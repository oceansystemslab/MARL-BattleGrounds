"""Check durable recording checkpoints and learner-linked rewind boundaries.

Short real capture packets provide the replay layout. Host-focused cases check
CSV prefix identity, unfinished replay restoration, bounded copies, pass/schema
rejection and failure ordering without a long training run. Learner state remains
caller-owned; the token must be saved with its matching numerical checkpoint.
Child preflight must reject every corrupt parent boundary before output exists,
prepare each prefix once, and close owned streams if child setup fails.
"""

import csv
import json
import os
from _hashlib import HASH
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation import recording_checkpoint
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, System
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.replay_io import ReplayLoadError
from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)


@pytest.fixture(scope="module")
def episode_infos() -> tuple[EpisodeInfo, ...]:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("priest",), team_b_roster=("priest",), max_steps=3
    )
    env = marl_bgs.make(
        "tdm", env_config=config, metrics="priority", replay_episodes=(1,)
    )
    _, state = env.reset(jax.random.key(7))
    zero = jnp.zeros(10, jnp.int32)
    step = jax.jit(env.step)
    records: list[EpisodeInfo] = []
    for index in range(3):
        _, state, _, _, info = cast(
            tuple[Any, ...],
            step(jax.random.key(index + 8), state, Action(zero, zero, zero)),
        )
        records.append(cast(EpisodeInfo, jax.device_get(info)))
    assert not bool(records[0].completed) and bool(records[-1].completed)
    return tuple(records)


def _never_call(*_args: object) -> object:
    raise AssertionError("checkpointing called a researcher method")


def _systems() -> dict[str, object]:
    system = System(
        "checkpoint router", _never_call, components=({"name": "a"}, {"name": "b"})
    )
    return {"team_a": system, "team_b": system}


def _trace(info: EpisodeInfo, decision: int | None = None) -> PolicyTrace:
    index = int(info.decision_step) if decision is None else decision
    return PolicyTrace(
        info.episode_id,
        cast(jax.Array, np.asarray(index, np.int32)),
        cast(jax.Array, np.asarray(True)),
        cast(jax.Array, np.where(info.active_mask, index % 2, -1).astype(np.int32)),
    )


def _files(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def _manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((path / "run_details.json").read_bytes()))


def _bundle(path: Path, token: dict[str, object]) -> Path:
    return path / "recording_checkpoints" / str(token["checkpoint_id"])


def _rewrite_bundle(
    path: Path, token: dict[str, object], updates: dict[str, object]
) -> dict[str, object]:
    from marl_battlegrounds.evaluation.run_writer import (
        _json_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    target = _bundle(path, token) / "checkpoint.json"
    content = json.loads(target.read_bytes())
    content.update(updates)
    payload = _json_bytes(content)
    target.write_bytes(payload)
    return {**token, "checkpoint_sha256": sha256(payload).hexdigest()}


def _sampled_start(info: EpisodeInfo) -> tuple[EnvConfig, EpisodeStartRecords]:
    team_a, team_b = canonical_tournament_rosters()
    source = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=team_a, team_b_roster=team_b, max_steps=3
    )
    bank_id, _, _ = ordered_source_bank_identity(source)
    words = np.frombuffer(bytes.fromhex(bank_id), dtype=">u4").astype(np.uint32)
    start = EpisodeStartRecords(
        info.episode_id,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(words),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(-1, jnp.int32),
        jnp.asarray(True),
        jnp.asarray(False),
        jnp.asarray(True),
        info.config.agent_profile.class_ids,
    )
    return source, start


@pytest.mark.parametrize("route", ["ordinary", "token"])
@pytest.mark.parametrize(
    "change",
    [
        "classes",
        "omit_classes",
        "omit_both",
        "null",
        "bool",
        "sentinel",
        "hole",
        "spawn",
        "resolved",
    ],
)
def test_saved_roster_tampering_rejects_both_resume_routes_before_mutation(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...], route: str, change: str
) -> None:
    from marl_battlegrounds.evaluation.run_writer import (
        _json_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path, phase="training") as writer:
        writer.register_episodes(start, source_configs=source)
        writer.write(
            episode_infos[0]._replace(replay=None, episode_start_records=start)
        )
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        path = writer.run_dir
    location = path if route == "ordinary" else _bundle(path, token)
    manifest = _manifest(location)
    saved = manifest["passes"][token["pass_key"]]["episode_starts"]["1"]
    if change == "classes":
        saved["source_class_ids"][0] = 1
    elif change in ("omit_classes", "omit_both"):
        del saved["source_class_ids"]
        if change == "omit_both":
            del saved["resolved_config_id"]
    elif change == "null":
        saved["source_class_ids"] = None
    elif change == "bool":
        saved["source_class_ids"][0] = True
    elif change == "sentinel":
        saved["source_class_ids"] = [-1] * 10
    elif change == "hole":
        saved["source_class_ids"][:3] = [0, 1, 0]
    elif change == "spawn":
        saved["spawn_locations"] = 1
    else:
        saved["resolved_config_id"] = manifest["source_banks"][
            saved["source_table_id"]
        ][0]
    payload = _json_bytes(manifest)
    (location / "run_details.json").write_bytes(payload)
    if route == "token":
        descriptor = json.loads((location / "checkpoint.json").read_bytes())
        descriptor["files"]["run_details.json"] = {
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
        token = _rewrite_bundle(path, token, descriptor)
    # An uncommitted suffix makes an early recovery call observably destructive.
    table = path / "episodes.csv"
    table.write_bytes(table.read_bytes() + b"uncommitted-suffix\n")
    before = _files(path)
    with pytest.raises(ValueError):
        RunWriter(
            resume_from=path,
            phase="training",
            recording_checkpoint=token if route == "token" else None,
        )
    assert _files(path) == before


def test_ordinary_resume_preserves_pending_roster_and_later_verifies(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path, phase="training") as writer:
        writer.register_episodes(start, source_configs=source)
        path = writer.run_dir
    with RunWriter(resume_from=path, phase="training") as resumed:
        assert resumed.has_pending_numerical_starts
        with pytest.raises(ValueError, match="start"):
            resumed.checkpoint_recording()
        resumed.write(
            episode_infos[0]._replace(replay=None, episode_start_records=start)
        )
        token = resumed.checkpoint_recording()
    with RunWriter(
        resume_from=path, phase="training", recording_checkpoint=token
    ) as restored:
        assert not restored.has_pending_numerical_starts
        saved = _manifest(path)["passes"][token["pass_key"]]["episode_starts"]["1"]
        assert saved["verification"] == "verified"
        assert saved["source_class_ids"] == np.asarray(start.source_class_ids).tolist()


def test_sampled_start_restore_reproduces_uninterrupted_episode_rows(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path, phase="training") as writer:
        writer.register_episodes(start, source_configs=source)
        writer.write(
            episode_infos[0]._replace(replay=None, episode_start_records=start)
        )
        token = writer.checkpoint_recording()
        path = writer.run_dir
        for info in episode_infos[1:]:
            writer.write(info._replace(replay=None))
        writer.flush()
        expected = {
            name: (path / name).read_bytes() for name in _manifest(path)["tables"]
        }
        expected_start = _manifest(path)["passes"][token["pass_key"]]["episode_starts"][
            "1"
        ]
    with RunWriter(
        resume_from=path, phase="training", recording_checkpoint=token
    ) as resumed:
        for info in episode_infos[1:]:
            resumed.write(info._replace(replay=None))
    assert {name: (path / name).read_bytes() for name in expected} == expected
    assert (
        _manifest(path)["passes"][token["pass_key"]]["episode_starts"]["1"]
        == expected_start
    )


def test_zero_boundary_token_is_small_and_does_not_save_learner(tmp_path: Path) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        path = writer.run_dir
        assert set(token) == {
            "schema_version",
            "run_id",
            "pass_key",
            "checkpoint_id",
            "checkpoint_sha256",
        }
        assert len(json.dumps(token)) < 512
        assert not writer.completed_episode_ids
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert not resumed.completed_episode_ids
        assert "recording_restore" not in _manifest(path)
        with pytest.raises(ValueError, match="pass"):
            resumed.start_pass(phase="validation", pass_id="later")


def test_rewind_live_replay_reproduces_completed_records(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
) -> None:
    systems = _systems()
    with RunWriter(
        tmp_path, phase="training", policies=systems, buffer_size=1
    ) as writer:
        writer.write(episode_infos[0], policy_trace=_trace(episode_infos[0]))
        token = writer.checkpoint_recording()
        path = writer.run_dir
        bundle = json.loads((_bundle(path, token) / "checkpoint.json").read_bytes())
        assert bundle["open_replays"][0]["count"] == 1
        for info in episode_infos[1:]:
            writer.write(info, policy_trace=_trace(info))
        writer.flush()
        expected = {
            name: (path / name).read_bytes()
            for name in (
                "episodes.csv",
                "priority_metrics.csv",
                "policy_assignments.csv",
            )
        }
        replay_files = {p.name: p.read_bytes() for p in (path / "replays").iterdir()}
    with RunWriter(
        resume_from=path,
        recording_checkpoint=token,
        phase="training",
        policies=systems,
        buffer_size=1,
    ) as resumed:
        assert not resumed.completed_episode_ids
        for info in episode_infos[1:]:
            resumed.write(info, policy_trace=_trace(info))
        resumed.flush()
        assert resumed.completed_episode_ids == frozenset({1})
        assert expected == {name: (path / name).read_bytes() for name in expected}
        assert replay_files == {
            p.name: p.read_bytes() for p in (path / "replays").iterdir()
        }
        complete_token = resumed.checkpoint_recording()
    with RunWriter(
        resume_from=path,
        recording_checkpoint=complete_token,
        phase="training",
        policies=systems,
    ) as resumed:
        assert resumed.completed_episode_ids == frozenset({1})


def test_later_token_cannot_resurrect_removed_csv_suffix(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        early = writer.checkpoint_recording()
        writer.write(episode_infos[-1]._replace(replay=None))
        later = writer.checkpoint_recording()
        path = writer.run_dir
    with RunWriter(resume_from=path, recording_checkpoint=early, phase="training"):
        pass
    before = _files(path)
    with pytest.raises(ValueError, match="shorter"):
        RunWriter(resume_from=path, recording_checkpoint=later, phase="training")
    assert _files(path) == before


@pytest.mark.parametrize(
    "change", ["extra", "boolean", "path", "digest", "run", "pass"]
)
def test_bad_token_leaves_existing_run_unchanged(tmp_path: Path, change: str) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        path = writer.run_dir
    token = {
        **token,
        **{
            "extra": {"other": 1},
            "boolean": {"schema_version": True},
            "path": {"checkpoint_id": "../escape"},
            "digest": {"checkpoint_sha256": "0" * 64},
            "run": {"run_id": "another"},
            "pass": {"pass_key": "another"},
        }[change],
    }
    before = _files(path)
    with pytest.raises(ValueError):
        RunWriter(resume_from=path, recording_checkpoint=token, phase="training")
    assert _files(path) == before


def test_same_length_changed_csv_is_not_the_saved_prefix(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        path = writer.run_dir
    table = path / "episodes.csv"
    content = table.read_bytes()
    table.write_bytes(content.replace(b"training", b"Training", 1))
    before = _files(path)
    with pytest.raises(ValueError, match="prefix differs"):
        RunWriter(resume_from=path, recording_checkpoint=token, phase="training")
    assert _files(path) == before


def test_csv_prefix_parser_counts_quoted_newlines_as_one_row(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    with RunWriter(
        tmp_path, phase="training", checkpoint_id="line one\nline two"
    ) as writer:
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        path = writer.run_dir
    with RunWriter(
        resume_from=path,
        recording_checkpoint=token,
        phase="training",
        checkpoint_id="line one\nline two",
    ) as resumed:
        assert resumed.completed_episode_ids == frozenset({1})
        with (path / "episodes.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 1


def test_hashes_are_lazy_and_updated_without_rescanning(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        assert writer._checkpoint_hashes is None  # pyright: ignore[reportPrivateUsage]
        writer.checkpoint_recording()
        original = recording_checkpoint._hash_prefix  # pyright: ignore[reportPrivateUsage]
        paths: list[str] = []

        def counted(path: Path, size: int) -> HASH:
            paths.append(path.name)
            return original(path, size)

        monkeypatch.setattr(recording_checkpoint, "_hash_prefix", counted)
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        assert not any(name.endswith(".csv") for name in paths)
        path = writer.run_dir
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert resumed.completed_episode_ids == frozenset({1})


def test_restore_marker_precedes_truncation_and_retries_idempotently(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        writer.write(episode_infos[-1]._replace(replay=None))
        writer.flush()
        path = writer.run_dir
    original = RunWriter._recover_tables  # pyright: ignore[reportPrivateUsage]

    def interrupted(writer: RunWriter) -> None:
        assert _manifest(path)["recording_restore"]["token"] == token
        raise OSError("injected before truncate")

    with monkeypatch.context() as patch:
        patch.setattr(RunWriter, "_recover_tables", interrupted)
        with pytest.raises(OSError, match="injected"):
            RunWriter(resume_from=path, recording_checkpoint=token, phase="training")
    before = _files(path)
    with pytest.raises(ValueError, match="token"):
        RunWriter(resume_from=path, phase="training")
    assert _files(path) == before
    assert RunWriter._recover_tables is original  # pyright: ignore[reportPrivateUsage]
    for _ in range(2):
        with RunWriter(
            resume_from=path, recording_checkpoint=token, phase="training"
        ) as resumed:
            assert not resumed.completed_episode_ids
            assert "recording_restore" not in _manifest(path)


def test_partial_tables_are_checked_before_any_rewind(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        path = writer.run_dir
    with (path / "episodes.csv").open("ab") as stream:
        stream.write(b"uncommitted suffix")
    (path / "priority_metrics.csv").write_bytes(b"broken")
    before = _files(path)
    with pytest.raises(ValueError):
        RunWriter(resume_from=path, recording_checkpoint=token, phase="training")
    assert _files(path) == before


def test_checkpoint_rejects_other_passes_before_output(tmp_path: Path) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.start_pass(phase="validation", pass_id="v")
        writer.flush()
        before = _files(writer.run_dir)
        with pytest.raises(ValueError, match="dedicated"):
            writer.checkpoint_recording()
        assert _files(writer.run_dir) == before


@pytest.mark.parametrize("change", ["layout", "count", "extra"])
def test_open_replay_semantics_checked_even_with_matching_bundle_hash(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    change: str,
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.write(episode_infos[0])
        token = writer.checkpoint_recording()
        path = writer.run_dir
    content = json.loads((_bundle(path, token) / "checkpoint.json").read_bytes())
    if change == "layout":
        content["open_replays"][0]["layout"][0]["dtype"] = "<i8"
    elif change == "count":
        content["open_replays"][0]["count"] = 2
    else:
        content["files"]["extra.json"] = {"bytes": 0, "sha256": sha256(b"").hexdigest()}
    bad = _rewrite_bundle(path, token, content)
    before = _files(path)
    with pytest.raises((ValueError, EOFError)):
        RunWriter(resume_from=path, recording_checkpoint=bad, phase="training")
    assert _files(path) == before


@pytest.mark.parametrize("failure", ["copy", "rename", "root_fsync"])
def test_failed_later_bundle_keeps_prior_token_usable(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    writer = RunWriter(tmp_path, phase="training")
    writer.write(episode_infos[0])
    token = writer.checkpoint_recording()
    path = writer.run_dir
    writer.write(episode_infos[1])
    original_copy = ReplayCollector.checkpoint_streams
    original_rename = Path.rename
    original_sync = recording_checkpoint._sync_directory  # pyright: ignore[reportPrivateUsage]

    def bad_copy(
        collector: ReplayCollector, directory: Path
    ) -> list[dict[str, object]]:
        original_copy(collector, directory)
        raise OSError("injected after prefix copy")

    def bad_rename(source: Path, target: str | os.PathLike[str]) -> Path:
        original_rename(source, target)
        raise OSError("injected after bundle rename")

    def bad_sync(directory: Path) -> None:
        original_sync(directory)
        if directory.name == "recording_checkpoints":
            raise OSError("injected after parent fsync")

    with monkeypatch.context() as patch:
        if failure == "copy":
            patch.setattr(ReplayCollector, "checkpoint_streams", bad_copy)
        elif failure == "rename":
            patch.setattr(Path, "rename", bad_rename)
        else:
            patch.setattr(recording_checkpoint, "_sync_directory", bad_sync)
        with pytest.raises(OSError, match="injected"):
            writer.checkpoint_recording()
        with pytest.raises(RuntimeError, match="failed"):
            writer.flush()
    writer.close()
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        resumed.write(episode_infos[1])
        resumed.write(episode_infos[2])
        resumed.flush()
        assert resumed.completed_episode_ids == frozenset({1})


@pytest.mark.parametrize("after_clear", [False, True])
def test_failure_around_marker_clear_can_restore_same_token_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_clear: bool,
) -> None:
    from marl_battlegrounds.evaluation import run_writer

    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        path = writer.run_dir
    original = run_writer._atomic_json  # pyright: ignore[reportPrivateUsage]

    def fail_clear(target: Path, value: object) -> None:
        payload = cast(dict[str, object], value)
        if "recording_restore" not in payload:
            if after_clear:
                original(target, value)
            raise OSError("injected marker clear failure")
        original(target, value)

    with monkeypatch.context() as patch:
        patch.setattr(run_writer, "_atomic_json", fail_clear)
        with pytest.raises(OSError, match="injected"):
            RunWriter(resume_from=path, recording_checkpoint=token, phase="training")
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert not resumed.completed_episode_ids
        assert "recording_restore" not in _manifest(path)


def test_wrong_source_content_is_rejected_even_with_valid_bundle_digests(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
) -> None:
    from marl_battlegrounds.evaluation.recording_identity import (
        ordered_source_bank_identity,
    )
    from marl_battlegrounds.evaluation.run_writer import (
        _json_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        path = writer.run_dir
    bank_id, references, content = ordered_source_bank_identity(episode_infos[0].config)
    bundle = _bundle(path, token)
    manifest = json.loads((bundle / "run_details.json").read_bytes())
    manifest["configurations"] = content
    manifest["source_banks"] = {bank_id: references + references}
    payload = _json_bytes(manifest)
    (bundle / "run_details.json").write_bytes(payload)
    descriptor = json.loads((bundle / "checkpoint.json").read_bytes())
    descriptor["files"]["run_details.json"] = {
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    bad = _rewrite_bundle(path, token, descriptor)
    before = _files(path)
    with pytest.raises(ValueError, match="content/order"):
        RunWriter(resume_from=path, recording_checkpoint=bad, phase="training")
    assert _files(path) == before


def test_future_dictionary_schedule_remains_a_declaration_at_checkpoint(
    tmp_path: Path,
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.register_episodes([{"episode_id": 12, "configuration_digest": "1" * 64}])
        token = writer.checkpoint_recording()
        path = writer.run_dir
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert not resumed.completed_episode_ids


def test_restored_collector_spools_new_selected_episodes_on_disk(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        token = writer.checkpoint_recording()
        path = writer.run_dir
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        resumed.write(episode_infos[0])
        next_token = resumed.checkpoint_recording()
        bundle = json.loads(
            (_bundle(path, next_token) / "checkpoint.json").read_bytes()
        )
        assert len(bundle["open_replays"]) == 1
        assert bundle["open_replays"][0]["count"] == 1


def test_snapshot_copies_large_prefix_and_restores_append_position(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
) -> None:
    packets = cast(ReplayPackets, episode_infos[0].replay)
    with RunWriter(tmp_path, phase="training") as writer:
        writer.write_replay(packets)
        collector = writer._collector  # pyright: ignore[reportPrivateUsage]
        assert collector is not None
        episode = collector._episodes[1]  # pyright: ignore[reportPrivateUsage]
        assert episode.stream is not None
        index = 1
        while episode.stream.tell() <= recording_checkpoint.COPY_BLOCK_BYTES:
            writer.write_replay(
                packets._replace(
                    transition_index=cast(jax.Array, np.asarray([index], np.int32)),
                    initial=cast(jax.Array, np.asarray([False])),
                )
            )
            index += 1
        position = episode.stream.tell()
        token = writer.checkpoint_recording()
        assert episode.stream.tell() == position
        writer.write_replay(
            packets._replace(
                transition_index=cast(jax.Array, np.asarray([index], np.int32)),
                initial=cast(jax.Array, np.asarray([False])),
            )
        )
        assert episode.stream.tell() > position
        path = writer.run_dir
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert resumed._collector is not None  # pyright: ignore[reportPrivateUsage]
        restored = resumed._collector._episodes[1]  # pyright: ignore[reportPrivateUsage]
        assert restored.count == index
        assert restored.stream is not None and restored.stream.tell() == position


def test_restore_hashes_each_replay_prefix_once(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with RunWriter(tmp_path, phase="training") as writer:
        writer.write(episode_infos[0])
        token = writer.checkpoint_recording()
        path = writer.run_dir
    original = recording_checkpoint._hash_prefix  # pyright: ignore[reportPrivateUsage]
    hashed: list[str] = []

    def tracked(file: Path, size: int) -> HASH:
        hashed.append(file.name)
        return original(file, size)

    monkeypatch.setattr(recording_checkpoint, "_hash_prefix", tracked)
    with RunWriter(
        resume_from=path, recording_checkpoint=token, phase="training"
    ) as resumed:
        assert resumed._collector is not None  # pyright: ignore[reportPrivateUsage]
    assert hashed.count("1.npylog") == 1


@pytest.mark.parametrize("change", ["missing_config", "unfinished", "historical"])
def test_restore_checks_completed_config_ownership_before_mutation(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    change: str,
) -> None:
    from marl_battlegrounds.evaluation.run_writer import (
        _json_bytes,  # pyright: ignore[reportPrivateUsage]
    )

    with RunWriter(tmp_path, phase="training") as writer:
        writer.write(episode_infos[-1]._replace(replay=None))
        token = writer.checkpoint_recording()
        path = writer.run_dir
    bundle = _bundle(path, token)
    manifest = json.loads((bundle / "run_details.json").read_bytes())
    entry = manifest["passes"][token["pass_key"]]
    identifier = entry["completed_config_ids"]["1"]
    if change == "historical":
        del entry["completed_config_ids"]
    elif change == "missing_config":
        entry["completed_config_ids"]["1"] = "0" * 64
    else:
        entry["completed_config_ids"]["999"] = identifier
    payload = _json_bytes(manifest)
    (bundle / "run_details.json").write_bytes(payload)
    descriptor = json.loads((bundle / "checkpoint.json").read_bytes())
    descriptor["files"]["run_details.json"] = {
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }
    altered_token = _rewrite_bundle(path, token, descriptor)
    if change == "historical":
        with RunWriter(
            resume_from=path, recording_checkpoint=altered_token, phase="training"
        ) as resumed:
            assert resumed.completed_episode_ids == frozenset({1})
    else:
        before = _files(path)
        message = (
            "missing configuration" if change == "missing_config" else "unfinished"
        )
        with pytest.raises(ValueError, match=message):
            RunWriter(
                resume_from=path, recording_checkpoint=altered_token, phase="training"
            )
        assert _files(path) == before


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_syncs_new_output_ancestors_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    output = tmp_path / "new-output" / "seed-42"
    assert not output.parent.exists()
    writer = RunWriter(output, phase="training")
    directory = writer.run_dir.resolve()
    expected = [directory, *directory.parents]
    synced: list[Path] = []
    original = recording_checkpoint._sync_directory  # pyright: ignore[reportPrivateUsage]

    def observe(path: Path) -> None:
        original(path)
        synced.append(path)
        if path in expected:
            assert list((directory / "recording_checkpoints").glob("*/checkpoint.json"))
        if fail and path == output.parent:
            raise OSError("injected output ancestor sync failure")

    token: dict[str, object] | None = None
    with monkeypatch.context() as patch:
        patch.setattr(recording_checkpoint, "_sync_directory", observe)
        if fail:
            with pytest.raises(OSError, match="ancestor sync"):
                token = writer.checkpoint_recording()
            assert token is None
            visited = expected[: expected.index(output.parent) + 1]
            assert synced[-len(visited) :] == visited
            with pytest.raises(RuntimeError, match="failed"):
                writer.checkpoint_recording()
        else:
            token = writer.checkpoint_recording()
            assert token is not None
            assert synced[-len(expected) :] == expected
    writer.close()


def test_child_recording_keeps_prefix_and_resumes_without_parent_tables(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...]
) -> None:
    from marl_battlegrounds.evaluation.replay_io import load_replay

    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    before = _files(parent_path)
    with RunWriter(tmp_path / "child", phase="training") as child:
        cost = recording_checkpoint.fork_recording(
            child, parent_path, token, episode_ids=[1]
        )
        child_path = child.run_dir
        assert child.run_id != token["run_id"]
        assert cast(int, cost["copied_prefix_bytes"]) > 0
        assert cost["inherited_episode_ids"] == [1]
        assert not (child_path / "episodes.csv").exists()
        child.write(episode_infos[1])
        child_token = child.checkpoint_recording()
    assert _files(parent_path) == before
    with RunWriter(
        resume_from=child_path, phase="training", recording_checkpoint=child_token
    ) as resumed:
        resumed.write(episode_infos[2])
    details = _manifest(child_path)
    entry = details["passes"][child_token["pass_key"]]
    replay = load_replay(child_path / entry["replays"]["1"]["path"]).replay
    assert replay.header.context.identity.run_id == token["run_id"]
    assert details["recording_ancestry"]["1"]["prefix_packets"] >= 1
    assert len(entry["completed_episode_ids"]) == 1
    assert _files(parent_path) == before
    with RunWriter(tmp_path / "reference", phase="training") as reference:
        reference.register_episodes(start, source_configs=source)
        reference.write(episode_infos[0]._replace(episode_start_records=start))
        for info in episode_infos[1:]:
            reference.write(info)
        reference_path = reference.run_dir
    reference_details = _manifest(reference_path)
    reference_pass = next(iter(reference_details["passes"].values()))
    expected = load_replay(
        reference_path / reference_pass["replays"]["1"]["path"]
    ).replay
    origin = replay.header.context.identity.run_id
    reference_id = expected.header.context.identity.run_id
    # Run-scoped IDs differ; every captured physical value and action must match.
    for observed, wanted in (
        (replay.frames, expected.frames),
        (replay.transitions, expected.transitions),
    ):
        assert len(observed) == len(wanted)
        for actual, target in zip(observed, wanted, strict=True):
            assert actual.model_dump(mode="json") == json.loads(
                target.model_dump_json().replace(reference_id, origin)
            )


@pytest.mark.parametrize("fault", ["digest", "missing_game", "duplicate_game"])
def test_recording_child_rejects_bad_parent_without_mutating_it(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...], fault: str
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    before = _files(parent_path)
    if fault == "digest":
        token = {**token, "checkpoint_sha256": "0" * 64}
    games = (
        [2] if fault == "missing_game" else [1, 1] if fault == "duplicate_game" else [1]
    )
    with RunWriter(tmp_path / "child", phase="training") as child:
        child_before = _files(child.run_dir)
        with pytest.raises(ValueError):
            recording_checkpoint.fork_recording(
                child, parent_path, token, episode_ids=games
            )
        assert _files(child.run_dir) == child_before
    assert _files(parent_path) == before


@pytest.mark.parametrize(
    "fault",
    ["token", "bundle", "prefix_bytes", "prefix_count", "csv", "completed_replay"],
)
def test_recording_fork_preflight_rejects_corruption_before_child_output(
    tmp_path: Path, episode_infos: tuple[EpisodeInfo, ...], fault: str
) -> None:
    source, start = _sampled_start(episode_infos[0])
    completed = fault in {"csv", "completed_replay"}
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        if completed:
            for info in episode_infos[1:]:
                parent.write(info)
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    bundle_path = _bundle(parent_path, token)
    bundle = json.loads((bundle_path / "checkpoint.json").read_bytes())
    if fault == "token":
        token = {**token, "checkpoint_sha256": "0" * 64}
    elif fault == "bundle":
        token = _rewrite_bundle(parent_path, token, {"schema_version": True})
    elif fault == "prefix_bytes":
        prefix = bundle_path / bundle["open_replays"][0]["path"]
        raw = prefix.read_bytes()
        prefix.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif fault == "prefix_count":
        bundle["open_replays"][0]["count"] += 1
        token = _rewrite_bundle(parent_path, token, bundle)
    elif fault == "csv":
        table = parent_path / "episodes.csv"
        table.write_bytes(table.read_bytes().replace(b"training", b"Training", 1))
    else:
        entry = _manifest(parent_path)["passes"][token["pass_key"]]
        replay = parent_path / entry["replays"]["1"]["path"]
        raw = replay.read_bytes()
        replay.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    before = _files(parent_path)
    child_root = tmp_path / "child"
    with (
        pytest.raises((ValueError, OSError, EOFError, ReplayLoadError)),
        ExitStack() as stack,
    ):
        prepared = recording_checkpoint.prepare_recording_fork(
            parent_path, token, episode_ids=[] if completed else [1], policies=None
        )
        stack.callback(prepared.close)
        child_root.mkdir()
        with RunWriter(child_root, phase="training") as child:
            recording_checkpoint.attach_recording_fork(child, prepared)
    assert not child_root.exists()
    assert _files(parent_path) == before


def test_recording_fork_prepares_prefix_once_and_transfers_stream_ownership(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    original_hash = recording_checkpoint._hash_prefix  # pyright: ignore[reportPrivateUsage]
    original_load = recording_checkpoint._load_packet  # pyright: ignore[reportPrivateUsage]
    hashed: list[str] = []
    decoded: list[ReplayPackets] = []

    def hash_once(path: Path, size: int) -> HASH:
        hashed.append(path.name)
        return original_hash(path, size)

    def decode_once(stream: BinaryIO, template: ReplayPackets) -> ReplayPackets:
        packet = original_load(stream, template)
        decoded.append(packet)
        return packet

    monkeypatch.setattr(recording_checkpoint, "_hash_prefix", hash_once)
    monkeypatch.setattr(recording_checkpoint, "_load_packet", decode_once)
    prepared = recording_checkpoint.prepare_recording_fork(
        parent_path, token, episode_ids=[1], policies=None
    )
    collector = prepared.collector
    assert collector is not None
    stream = collector._episodes[1].stream  # pyright: ignore[reportPrivateUsage]
    assert stream is not None and not stream.closed
    expected_packets = cast(int, prepared.costs["copied_prefix_packets"])
    assert len(decoded) == expected_packets
    assert hashed.count("1.npylog") == 1
    with RunWriter(tmp_path / "child", phase="training") as child:
        with monkeypatch.context() as patch:
            patch.setattr(recording_checkpoint, "_hash_prefix", _never_call)
            patch.setattr(recording_checkpoint, "_load_packet", _never_call)
            cost = recording_checkpoint.attach_recording_fork(child, prepared)
        assert child._collector is collector  # pyright: ignore[reportPrivateUsage]
        assert prepared.collector is None
        prepared.close()
        assert not stream.closed
        assert cost["checked_open_prefix_bytes"] == cost["copied_prefix_bytes"]
        for info in episode_infos[1:]:
            child.write(info)
    assert stream.closed
    prepared.close()


@pytest.mark.parametrize("failure", ["mkdir", "lock", "writer", "attach"])
def test_recording_fork_setup_failure_closes_prepared_streams(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from marl_battlegrounds.training._run_io import run_lock

    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    before = _files(parent_path)
    prepared = recording_checkpoint.prepare_recording_fork(
        parent_path, token, episode_ids=[1], policies=None
    )
    assert prepared.collector is not None
    stream = prepared.collector._episodes[1].stream  # pyright: ignore[reportPrivateUsage]
    assert stream is not None and not stream.closed
    child_root = tmp_path / "child"
    original_mkdir = Path.mkdir

    def mkdir(
        path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if path == child_root:
            raise OSError("injected child mkdir failure")
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected child setup failure")

    with pytest.raises((OSError, RuntimeError)), ExitStack() as stack:
        stack.callback(prepared.close)
        if failure == "mkdir":
            monkeypatch.setattr(Path, "mkdir", mkdir)
        child_root.mkdir()
        stack.enter_context(run_lock(child_root))
        if failure == "lock":
            stack.enter_context(run_lock(child_root))
        if failure == "writer":
            monkeypatch.setattr(RunWriter, "__init__", fail)
        child = stack.enter_context(
            RunWriter(child_root / "episodes", phase="training")
        )
        if failure == "attach":
            from marl_battlegrounds.evaluation import run_writer as writer_module

            monkeypatch.setattr(writer_module, "_atomic_json", fail)
        recording_checkpoint.attach_recording_fork(child, prepared)
    assert stream.closed and prepared.collector is None
    assert _files(parent_path) == before
    prepared.close()


def test_recording_fork_checks_unselected_prefix_without_copying_it(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    monkeypatch.setattr(recording_checkpoint, "TemporaryFile", _never_call)
    prepared = recording_checkpoint.prepare_recording_fork(
        parent_path, token, episode_ids=[], policies=None
    )
    try:
        assert prepared.collector is not None
        assert prepared.collector.pending_episode_ids == frozenset()
        assert prepared.costs["copied_prefix_bytes"] == 0
        assert cast(int, prepared.costs["checked_open_prefix_bytes"]) > 0
    finally:
        prepared.close()
    bundle = json.loads((_bundle(parent_path, token) / "checkpoint.json").read_bytes())
    bundle["open_replays"][0]["count"] += 1
    bad = _rewrite_bundle(parent_path, token, bundle)
    with pytest.raises((ValueError, EOFError)):
        recording_checkpoint.prepare_recording_fork(
            parent_path, bad, episode_ids=[], policies=None
        )


def test_recording_fork_decode_failure_closes_its_new_stream(
    tmp_path: Path,
    episode_infos: tuple[EpisodeInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, start = _sampled_start(episode_infos[0])
    with RunWriter(tmp_path / "parent", phase="training") as parent:
        parent.register_episodes(start, source_configs=source)
        parent.write(episode_infos[0]._replace(episode_start_records=start))
        token = parent.checkpoint_recording()
        parent_path = parent.run_dir
    bundle = json.loads((_bundle(parent_path, token) / "checkpoint.json").read_bytes())
    bundle["open_replays"][0]["count"] += 1
    token = _rewrite_bundle(parent_path, token, bundle)
    temporary_file = recording_checkpoint.TemporaryFile
    streams: list[BinaryIO] = []

    def opened() -> BinaryIO:
        stream = temporary_file()
        streams.append(stream)
        return stream

    monkeypatch.setattr(recording_checkpoint, "TemporaryFile", opened)
    with pytest.raises((ValueError, EOFError)):
        recording_checkpoint.prepare_recording_fork(
            parent_path, token, episode_ids=[1], policies=None
        )
    assert len(streams) == 1 and streams[0].closed
