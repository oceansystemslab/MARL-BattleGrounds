"""Durable CSV runs from real episode completions, including interrupted writes."""

import csv
import json
import os
import stat
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds import make
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EnvironmentState, EpisodeInfo
from marl_battlegrounds.evaluation import run_writer
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, RunWriter
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


class _Episodes(NamedTuple):
    full: tuple[EpisodeInfo, EpisodeInfo, EpisodeInfo]
    priority: EpisodeInfo
    none: EpisodeInfo
    chunks: tuple[EpisodeInfo, EpisodeInfo]


def _config(*, alternate: bool = False, max_steps: int = 1) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=int(alternate),
        team_a_roster=("hunter",) if alternate else ("mage", "priest"),
        team_b_roster=("priest", "mage") if alternate else ("warrior",),
        score_threshold=12 if alternate else 20,
        max_steps=max_steps,
    )


def _idle(batch: int | None = None) -> Action:
    zeros = jnp.zeros((10,) if batch is None else (batch, 10), jnp.int32)
    return Action(zeros, zeros, zeros)


@pytest.fixture(scope="module")
def episodes() -> _Episodes:
    key = jax.random.key(1)

    def record(
        mode: Literal["none", "priority", "full"],
        episode_id: int,
        alternate: bool = False,
    ) -> EpisodeInfo:
        env = make("tdm", metrics=mode)
        _, initial = env.reset(key, _config(alternate=alternate), episode_id=episode_id)
        result = cast(
            tuple[object, EnvironmentState, object, object, EpisodeInfo],
            jax.jit(env.step)(key, initial, _idle()),
        )
        assert bool(result[4].completed)
        return result[4]

    full = (record("full", 1), record("full", 2, True), record("full", 3))
    priority, none = record("priority", 4), record("none", 5)
    env = make("tdm", num_envs=2)

    def stack(*values: Array) -> Array:
        return jnp.stack(values)

    config = jax.tree.map(stack, _config(), _config(alternate=True, max_steps=2))
    _, initial = env.reset(key, config, episode_id=jnp.asarray((101, 201), jnp.int32))

    def advance(
        state: EnvironmentState, step_key: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        _, successor, _, _, info = env.step(step_key, state, _idle(2))
        return successor, info

    def scan(state: EnvironmentState) -> tuple[EnvironmentState, EpisodeInfo]:
        return jax.lax.scan(advance, state, jax.random.split(key, 2))

    finished, first = cast(tuple[EnvironmentState, EpisodeInfo], jax.jit(scan)(initial))
    _, reset_state = env.reset(
        key,
        _config(alternate=True),
        episode_id=jnp.asarray((102, 999), jnp.int32),
        state=finished,
        reset_mask=jnp.asarray((True, False)),
    )
    _, second = cast(
        tuple[EnvironmentState, EpisodeInfo], jax.jit(advance)(reset_state, key)
    )
    np.testing.assert_array_equal(first.completed, ((True, False), (False, True)))
    np.testing.assert_array_equal(second.completed, (True, False))
    return _Episodes(full, priority, none, (first, second))


def _details(run_dir: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((run_dir / "run_details.json").read_bytes()))


def _pass(
    run_dir: Path, phase: str = "evaluation", pass_id: str = "1"
) -> dict[str, Any]:
    found = [
        entry
        for entry in _details(run_dir)["passes"].values()
        if entry["phase"] == phase and entry["pass_id"] == pass_id
    ]
    assert len(found) == 1
    return found[0]


def _table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        rows = list(reader)
    assert all(None not in row and None not in row.values() for row in rows)
    return list(reader.fieldnames), rows


def test_new_run_directories_are_unique_and_existing_run_is_not_an_output_root(
    tmp_path: Path,
) -> None:
    with RunWriter(tmp_path) as first, RunWriter(tmp_path) as second:
        assert first.run_dir.parent == second.run_dir.parent == tmp_path
        assert first.run_dir != second.run_dir and first.run_id != second.run_id
        first_bytes = first.paths["run_details"].read_bytes()
        with pytest.raises(ValueError, match="existing run"):
            RunWriter(first.run_dir)
        assert first.paths["run_details"].read_bytes() == first_bytes
        assert set(first.paths) == {"run_details"}


def test_csv_headers_full_priority_copy_and_missing_values_survive_roster_changes(
    tmp_path: Path, episodes: _Episodes
) -> None:
    policies: dict[str, object] = {"team_a": 'model, "checkpoint 7"', "team_b": "BETA"}
    with RunWriter(tmp_path, policies=policies, checkpoint_id="weights-7") as writer:
        writer.write(episodes.full[0])
        writer.write(episodes.full[1])
        writer.flush()
        priority_header, priority = _table(writer.paths["priority_metrics"])
        full_header, full = _table(writer.paths["full_metrics"])
        assert len(IDENTITY_COLUMNS) == 30
        assert priority_header == [*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES]
        assert full_header == [*IDENTITY_COLUMNS, *FULL_METRIC_NAMES]
        assert len(priority_header) == 30 + 26 and len(full_header) == 30 + 11_158
        assert len(priority) == len(full) == 2
        for small, complete in zip(priority, full, strict=True):
            assert {name: complete[name] for name in priority_header} == small
            assert complete["run_id"] == writer.run_id
            assert complete["team_a_policy"] == policies["team_a"]
            assert complete["team_b_policy"] == "BETA"
            assert complete["checkpoint_id"] == "weights-7"
            assert complete["seed_id"] == complete["map_id"] == ""
            assert float(complete["episode_length"]) == 1
        assert float(full[0]["agent_0_healing_done"]) == 0
        assert float(full[0]["agent_1_damage_done"]) == 0
        assert full[0]["agent_2_return"] == ""
        assert full[0]["agent_0_excess_healing"] == ""
        assert full[0]["agent_0_damage_done_fraction"] == ""
        assert full[1]["agent_1_return"] == ""
        assert full[0]["config_id"] != full[1]["config_id"]
        details = _details(writer.run_dir)
        assert details["metric_schema_id"] == METRIC_SCHEMA_ID
        assert details["metric_schema_version"] == METRIC_SCHEMA_VERSION
        for row, info in zip(full, episodes.full[:2], strict=True):
            config = details["configurations"][row["config_id"]]
            np.testing.assert_array_equal(
                config["agent_profile"]["class_ids"], info.class_ids
            )
            for slot in range(10):
                assert int(row[f"agent_{slot}_class_id"]) == int(info.class_ids[slot])
                assert bool(int(row[f"agent_{slot}_active"])) == bool(
                    info.active_mask[slot]
                )


def test_sparse_tables_and_none_mode_do_not_create_unrequested_csvs(
    tmp_path: Path, episodes: _Episodes
) -> None:
    with RunWriter(tmp_path, buffer_size=1) as writer:
        writer.write(episodes.priority)
        assert set(writer.paths) == {"run_details", "priority_metrics"}
        writer.write(episodes.full[0])
        writer.write(episodes.none)
        assert len(_table(writer.paths["priority_metrics"])[1]) == 2
        assert len(_table(writer.paths["full_metrics"])[1]) == 1
        assert writer.completed_episode_ids == frozenset((1, 4, 5))
    with RunWriter(tmp_path) as disabled:
        disabled.write(episodes.none)
        disabled.flush()
        assert set(disabled.paths) == {"run_details"}
        assert not list(disabled.run_dir.glob("*.csv"))
        assert disabled.completed_episode_ids == frozenset((5,))


def test_chunk_writes_keep_every_completion_and_its_pre_reset_configuration(
    tmp_path: Path, episodes: _Episodes
) -> None:
    with RunWriter(tmp_path, buffer_size=2) as writer:
        writer.write(episodes.chunks[0])
        writer.write(episodes.chunks[1])
        writer.flush()
        _, rows = _table(writer.paths["priority_metrics"])
        assert [int(row["episode_id"]) for row in rows] == [101, 201, 102]
        assert [float(row["episode_length"]) for row in rows] == [1, 2, 1]
        assert [int(row["agent_0_class_id"]) for row in rows] == [1, 3, 3]
        assert [int(row["agent_1_active"]) for row in rows] == [1, 0, 0]
        details = _details(writer.run_dir)
        configs = [details["configurations"][row["config_id"]] for row in rows]
        assert [config["max_steps"] for config in configs] == [1, 2, 1]
        assert [config["team_deathmatch_score_threshold"] for config in configs] == [
            20,
            12,
            12,
        ]
        assert writer.completed_episode_ids == frozenset((101, 201, 102))


def test_buffer_threshold_flush_and_close_only_advertise_durable_ids(
    tmp_path: Path, episodes: _Episodes
) -> None:
    writer = RunWriter(tmp_path, buffer_size=2)
    writer.write(episodes.full[0])
    assert writer.completed_episode_ids == frozenset()
    assert _pass(writer.run_dir)["completed_episode_ids"] == []
    assert not (writer.run_dir / "priority_metrics.csv").exists()
    writer.write(episodes.full[1])
    assert writer.completed_episode_ids == frozenset((1, 2))
    prefix = writer.paths["priority_metrics"].read_bytes()
    writer.write(episodes.full[2])
    assert writer.paths["priority_metrics"].read_bytes() == prefix
    assert writer.completed_episode_ids == frozenset((1, 2))
    writer.close()
    assert writer.completed_episode_ids == frozenset((1, 2, 3))
    assert writer.paths["priority_metrics"].read_bytes().startswith(prefix)
    assert len(_table(writer.paths["priority_metrics"])[1]) == 3
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(episodes.full[0])


def test_shared_passes_keep_policy_checkpoint_identity_and_separate_episode_keys(
    tmp_path: Path, episodes: _Episodes
) -> None:
    policies: dict[str, object] = {"team_a": "learner", "team_b": "BETA"}
    with RunWriter(
        tmp_path, phase="training", pass_id="train", policies=policies
    ) as writer:
        writer.write(episodes.priority)
        writer.start_pass(
            phase="validation",
            pass_id="val-1",
            policies=policies,
            checkpoint_id="weights-7",
        )
        assert _pass(writer.run_dir, "training", "train")["completed_episode_ids"] == [
            4
        ]
        assert writer.completed_episode_ids == frozenset()
        writer.write(episodes.priority)
        writer.flush()
        _, rows = _table(writer.paths["priority_metrics"])
        assert [(row["phase"], row["pass_id"], row["episode_id"]) for row in rows] == [
            ("training", "train", "4"),
            ("validation", "val-1", "4"),
        ]
        training = _pass(writer.run_dir, "training", "train")
        validation = _pass(writer.run_dir, "validation", "val-1")
        assert (
            training["policy_state"] == "evolving" and training["checkpoint_id"] is None
        )
        assert (
            validation["policy_state"] == "frozen"
            and validation["checkpoint_id"] == "weights-7"
        )
        with pytest.raises(ValueError, match="identity differs"):
            writer.start_pass(
                phase="validation",
                pass_id="val-1",
                policies=policies,
                checkpoint_id="different",
            )


def test_duplicate_pending_and_durable_records_fail_instead_of_double_counting(
    tmp_path: Path, episodes: _Episodes
) -> None:
    for durable in (False, True):
        writer = RunWriter(tmp_path)
        writer.write(episodes.priority)
        if durable:
            writer.flush()
        with pytest.raises(ValueError, match="already written"):
            writer.write(episodes.priority)
        assert writer.completed_episode_ids == (
            frozenset((4,)) if durable else frozenset()
        )
        writer.close()
        with RunWriter(resume_from=writer.run_dir) as resumed:
            assert resumed.completed_episode_ids == (
                frozenset((4,)) if durable else frozenset()
            )
            if not durable:
                resumed.write(episodes.priority)
                resumed.flush()
            assert len(_table(resumed.paths["priority_metrics"])[1]) == 1


def test_registered_schedule_is_durable_and_supplies_truthful_row_identity(
    tmp_path: Path, episodes: _Episodes
) -> None:
    schedule: list[dict[str, object]] = [
        {
            "episode_id": 1,
            "seed_id": 701,
            "map_id": 0,
            "policies": {
                "team_a": {"name": "saved learner", "checkpoint_id": "hash-7"},
                "team_b": {"name": "BETA"},
            },
        }
    ]
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(schedule)
        assert _pass(writer.run_dir)["episodes"]["1"] == schedule[0]
        assert writer.completed_episode_ids == frozenset()
        assert not list(writer.run_dir.glob("*.csv"))
        writer.register_episodes(schedule)
        writer.write(episodes.full[0])
        writer.flush()
        _, rows = _table(writer.paths["priority_metrics"])
        assert len(rows) == 1
        assert rows[0]["seed_id"] == "701" and rows[0]["map_id"] == "0"
        assert rows[0]["team_a_policy"] == "saved learner"
        assert rows[0]["team_b_policy"] == "BETA"
    with RunWriter(resume_from=writer.run_dir) as resumed:
        resumed.register_episodes(schedule)
        with pytest.raises(ValueError, match="differs from the recorded schedule"):
            resumed.register_episodes([{**schedule[0], "seed_id": 702}])
        assert resumed.completed_episode_ids == frozenset((1,))
        assert _pass(resumed.run_dir)["episodes"]["1"] == schedule[0]


def test_resume_truncates_uncommitted_csv_suffixes_and_retries_once(
    tmp_path: Path, episodes: _Episodes
) -> None:
    with RunWriter(tmp_path, buffer_size=1) as writer:
        writer.write(episodes.full[0])
    original: dict[str, bytes] = {}
    for name in ("priority_metrics", "full_metrics"):
        path = writer.paths[name]
        original[name] = path.read_bytes()
        with path.open("ab") as stream:
            stream.write(b'"unterminated,uncommitted')
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == frozenset((1,))
        for name, payload in original.items():
            assert resumed.paths[name].read_bytes() == payload
        resumed.write(episodes.full[1])
        resumed.flush()
        for name in original:
            _, rows = _table(resumed.paths[name])
            assert [row["episode_id"] for row in rows] == ["1", "2"]
            table = _details(resumed.run_dir)["tables"][f"{name}.csv"]
            assert table["rows"] == 2
            assert table["durable_bytes"] == resumed.paths[name].stat().st_size


@pytest.mark.parametrize("damage", ("missing", "truncated", "schema"))
def test_resume_rejects_missing_or_corrupt_durable_artifacts(
    tmp_path: Path, episodes: _Episodes, damage: str
) -> None:
    with RunWriter(tmp_path, buffer_size=1) as writer:
        writer.write(episodes.priority)
    table = writer.paths["priority_metrics"]
    if damage == "missing":
        table.unlink()
    elif damage == "truncated":
        table.write_bytes(table.read_bytes()[:-1])
    else:
        details = _details(writer.run_dir)
        details["metric_schema_version"] = METRIC_SCHEMA_VERSION + 1
        writer.paths["run_details"].write_text(json.dumps(details))
        # Reject incompatible schemas before attempting interrupted-tail recovery.
        with table.open("ab") as stream:
            stream.write(b"unfinished row\n")
    before = {
        path.relative_to(writer.run_dir): path.read_bytes()
        for path in writer.run_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError):
        RunWriter(resume_from=writer.run_dir)
    assert before == {
        path.relative_to(writer.run_dir): path.read_bytes()
        for path in writer.run_dir.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("old_version", range(1, METRIC_SCHEMA_VERSION))
def test_old_scalar_schema_resume_preserves_unfinished_files(
    tmp_path: Path, episodes: _Episodes, old_version: int
) -> None:
    """Schema incompatibility is rejected before interrupted-tail recovery."""
    with RunWriter(tmp_path, buffer_size=1) as writer:
        writer.write(episodes.full[0])
    details = _details(writer.run_dir)
    details["metric_schema_version"] = old_version
    writer.paths["run_details"].write_text(json.dumps(details))
    for key in ("priority_metrics", "full_metrics"):
        with writer.paths[key].open("ab") as stream:
            stream.write(b'"unfinished old-schema row')
    before = {
        path.relative_to(writer.run_dir): path.read_bytes()
        for path in writer.run_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="start a new run"):
        RunWriter(resume_from=writer.run_dir)
    assert before == {
        path.relative_to(writer.run_dir): path.read_bytes()
        for path in writer.run_dir.rglob("*")
        if path.is_file()
    }


def test_resume_requires_matching_policy_identity_and_releases_failed_constructor_lock(
    tmp_path: Path, episodes: _Episodes
) -> None:
    policies: dict[str, object] = {"team_a": "trained-7", "team_b": "BETA"}
    with RunWriter(tmp_path, policies=policies, buffer_size=1) as writer:
        writer.write(episodes.priority)
    with pytest.raises(ValueError, match="identity differs"):
        RunWriter(
            resume_from=writer.run_dir,
            policies={"team_a": "different", "team_b": "BETA"},
        )
    with RunWriter(resume_from=writer.run_dir, policies=policies) as resumed:
        assert resumed.completed_episode_ids == frozenset((4,))


def test_live_writer_lock_prevents_stale_metadata_recovery_during_handoff(
    tmp_path: Path, episodes: _Episodes, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = RunWriter(tmp_path, buffer_size=1)
    owner.write(episodes.full[0])
    metadata_path = owner.paths["run_details"]
    read_bytes = Path.read_bytes
    raced = False

    def read_during_handoff(path: Path) -> bytes:
        nonlocal raced
        payload = read_bytes(path)
        if path == metadata_path and not raced:
            raced = True
            owner.write(episodes.full[1])
            owner.close()
        return payload

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", read_during_handoff)
            with pytest.raises(BlockingIOError):
                RunWriter(resume_from=owner.run_dir)
        assert not raced  # Lock rejection preceded any stale metadata read.
        owner.write(episodes.full[1])
    finally:
        owner.close()
    with RunWriter(resume_from=owner.run_dir) as resumed:
        assert resumed.completed_episode_ids == frozenset((1, 2))
        assert len(_table(resumed.paths["priority_metrics"])[1]) == 2


@pytest.mark.parametrize("fault", ("nonfinite", "shape"))
def test_invalid_measurement_write_fails_loudly_without_committing_partial_rows(
    tmp_path: Path, episodes: _Episodes, fault: str
) -> None:
    info = episodes.full[0]
    assert info.full is not None
    if fault == "nonfinite":
        malformed = info.full._replace(values=info.full.values.at[26].set(jnp.nan))
    else:
        malformed = info.full._replace(values=info.full.values[:-1])
    writer = RunWriter(tmp_path)
    with pytest.raises(ValueError, match=r"nonfinite|schema"):
        writer.write(info._replace(full=malformed))
    assert writer.completed_episode_ids == frozenset()
    assert _pass(writer.run_dir)["completed_episode_ids"] == []
    assert not list(writer.run_dir.glob("*.csv"))
    with pytest.raises(RuntimeError, match="failed"):
        writer.flush()
    writer.close()
    errors = (writer.run_dir / "failures.jsonl").read_text().splitlines()
    assert len(errors) == 1 and json.loads(errors[0])["type"] == "ValueError"


@pytest.mark.parametrize("fault", ("table_fsync", "manifest"))
def test_failed_flush_preserves_durable_boundary_and_allows_exactly_once_retry(
    tmp_path: Path, episodes: _Episodes, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer = RunWriter(tmp_path)
    writer.register_episodes(
        {"episode_id": episode, "seed_id": episode + 10} for episode in (1, 2)
    )
    writer.write(episodes.full[0])
    writer.flush()
    durable_metadata = writer.paths["run_details"].read_bytes()
    durable_tables = {
        name: path.read_bytes()
        for name, path in writer.paths.items()
        if name != "run_details"
    }
    writer.write(episodes.full[1])
    before_flush = run_writer._json_bytes(writer._details)  # pyright: ignore[reportPrivateUsage]
    # A pending reference exercises failure isolation without publishing a replay.
    pending_replay: dict[str, object] = {
        "path": "replays/pending.marlbg-replay.json",
        "bytes": 123,
    }
    writer._pending_replays["2"] = pending_replay  # pyright: ignore[reportPrivateUsage]
    fsync = os.fsync

    def fail_file_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected table fsync failure")
        fsync(descriptor)

    def fail_manifest(path: Path, value: object) -> None:
        del path
        candidate = cast(dict[str, Any], value)
        candidate_pass = next(iter(candidate["passes"].values()))
        assert candidate_pass["completed_episode_ids"] == [1, 2]
        assert candidate_pass["replays"] == {"2": pending_replay}
        assert all(table["rows"] == 2 for table in candidate["tables"].values())
        raise OSError("injected manifest failure")

    with monkeypatch.context() as patch:
        if fault == "table_fsync":
            patch.setattr(run_writer.os, "fsync", fail_file_sync)
        else:
            patch.setattr(run_writer, "_atomic_json", fail_manifest)
        with pytest.raises(OSError, match="injected"):
            writer.flush()
    assert writer.paths["run_details"].read_bytes() == durable_metadata
    assert run_writer._json_bytes(writer._details) == before_flush  # pyright: ignore[reportPrivateUsage]
    assert writer.completed_episode_ids == frozenset((1,))
    writer.close()
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == frozenset((1,))
        for name, payload in durable_tables.items():
            assert resumed.paths[name].read_bytes() == payload
        resumed.write(episodes.full[1])
        resumed.flush()
        assert resumed.completed_episode_ids == frozenset((1, 2))
        for name in durable_tables:
            assert [row["episode_id"] for row in _table(resumed.paths[name])[1]] == [
                "1",
                "2",
            ]
    assert "injected" in (writer.run_dir / "failures.jsonl").read_text()


def test_close_propagates_flush_failure_and_releases_the_lock(
    tmp_path: Path, episodes: _Episodes, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = RunWriter(tmp_path)
    writer.write(episodes.priority)

    def fail_manifest(path: Path, value: object) -> None:
        del path, value
        raise OSError("injected close failure")

    with monkeypatch.context() as patch:
        patch.setattr(run_writer, "_atomic_json", fail_manifest)
        with pytest.raises(OSError, match="injected close"):
            writer.close()
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == frozenset()
        resumed.write(episodes.priority)
        resumed.flush()
        assert resumed.completed_episode_ids == frozenset((4,))


@pytest.mark.parametrize("fault", ("none", "flush", "close"))
def test_context_failure_preserves_original_error_and_resumable_results(
    tmp_path: Path,
    episodes: _Episodes,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    writer = RunWriter(tmp_path)
    writer.write(episodes.full[0])
    writer.flush()
    writer.write(episodes.full[1])
    failure = RuntimeError("rating fit failed")

    def failed_manifest(path: Path, value: object) -> None:
        del path, value
        raise OSError("manifest storage failed")

    class FailedCleanup:
        def close(self) -> None:
            raise OSError("collector cleanup failed")

    with monkeypatch.context() as patch:
        if fault == "flush":
            patch.setattr(run_writer, "_atomic_json", failed_manifest)
        elif fault == "close":
            patch.setattr(writer, "_collector", FailedCleanup())
        with pytest.raises(RuntimeError, match="rating fit failed") as caught, writer:
            raise failure
    assert caught.value is failure
    failures = [
        json.loads(line)
        for line in (writer.run_dir / "failures.jsonl").read_text().splitlines()
    ]
    assert sum(row["message"] == "rating fit failed" for row in failures) == 1
    if fault != "none":
        detail = "storage failed" if fault == "flush" else "cleanup failed"
        assert detail in " ".join(failure.__notes__)
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == (
            frozenset((1,)) if fault == "flush" else frozenset((1, 2))
        )
        if fault == "flush":
            resumed.write(episodes.full[1])
            resumed.flush()
        for name in ("priority_metrics", "full_metrics"):
            assert [row["episode_id"] for row in _table(resumed.paths[name])[1]] == [
                "1",
                "2",
            ]


@pytest.mark.parametrize("flush_fails", (False, True))
def test_cleanup_without_body_error_records_first_failure_and_releases_lock(
    tmp_path: Path,
    episodes: _Episodes,
    monkeypatch: pytest.MonkeyPatch,
    flush_fails: bool,
) -> None:
    writer = RunWriter(tmp_path)
    writer.write(episodes.full[0])
    storage_error = OSError("manifest storage failed")
    cleanup_error = OSError("collector cleanup failed")

    def failed_manifest(path: Path, value: object) -> None:
        del path, value
        raise storage_error

    class FailedCleanup:
        def close(self) -> None:
            raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(writer, "_collector", FailedCleanup())
        if flush_fails:
            patch.setattr(run_writer, "_atomic_json", failed_manifest)
        with pytest.raises(OSError) as caught, writer:
            pass
    expected = storage_error if flush_fails else cleanup_error
    assert caught.value is expected
    failures = (writer.run_dir / "failures.jsonl").read_text().splitlines()
    assert len(failures) == 1
    assert json.loads(failures[0])["message"] == str(expected)
    if flush_fails:
        assert "collector cleanup failed" in " ".join(caught.value.__notes__)
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == (
            frozenset() if flush_fails else frozenset((1,))
        )


def test_directory_sync_failure_is_loud_and_resume_uses_surviving_consistent_boundary(
    tmp_path: Path, episodes: _Episodes, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = RunWriter(tmp_path)
    writer.write(episodes.full[0])
    writer.flush()
    writer.write(episodes.full[1])
    fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(run_writer.os, "fsync", fail_directory_sync)
        with pytest.raises(OSError, match="injected directory"):
            writer.flush()
    assert writer.completed_episode_ids == frozenset((1,))
    with pytest.raises(RuntimeError, match="failed"):
        writer.write(episodes.full[2])
    # Failed directory synchronization leaves publication uncertain, not rolled back.
    surviving = frozenset(_pass(writer.run_dir)["completed_episode_ids"])
    assert surviving in (frozenset((1,)), frozenset((1, 2)))
    writer.close()
    with RunWriter(resume_from=writer.run_dir) as resumed:
        assert resumed.completed_episode_ids == surviving
        details = _details(resumed.run_dir)
        for name in ("priority_metrics", "full_metrics"):
            _, rows = _table(resumed.paths[name])
            assert frozenset(int(row["episode_id"]) for row in rows) == surviving
            table = details["tables"][f"{name}.csv"]
            assert table["rows"] == len(rows)
            assert table["durable_bytes"] == resumed.paths[name].stat().st_size
        if 2 not in resumed.completed_episode_ids:
            resumed.write(episodes.full[1])
        resumed.write(episodes.full[2])
        resumed.flush()
        for name in ("priority_metrics", "full_metrics"):
            assert [row["episode_id"] for row in _table(resumed.paths[name])[1]] == [
                "1",
                "2",
                "3",
            ]
    assert "injected directory" in (writer.run_dir / "failures.jsonl").read_text()


@pytest.mark.parametrize("operation", ("append", "resume"))
def test_owned_table_symlink_never_modifies_an_unrelated_file(
    tmp_path: Path, episodes: _Episodes, operation: str
) -> None:
    unrelated = tmp_path / "unrelated.csv"
    original = b"unrelated,data\nkeep,exactly\n"
    unrelated.write_bytes(original)
    writer = RunWriter(tmp_path / "runs", buffer_size=1)
    table = writer.run_dir / "priority_metrics.csv"
    if operation == "resume":
        writer.write(episodes.priority)
        writer.close()
        table.unlink()
    table.symlink_to(unrelated)
    try:
        with pytest.raises((OSError, ValueError)):
            if operation == "resume":
                RunWriter(resume_from=writer.run_dir)
            else:
                writer.write(episodes.priority)
    finally:
        writer.close()
    assert unrelated.read_bytes() == original
