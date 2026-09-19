"""Check tracker source identity, empty registration and learner-data exclusion.

The writer must share the tracker's ordered source-bank identity, validate empty
start packets without flushing unrelated data, and discard final learning inputs
before a host transfer. Live tracker tests also check source/start ownership when
short games finish beside continuing games and curriculum configurations change.
Legacy nine-argument start producers retain absent roster declarations through
scalar, batch and time/batch JAX trees and empty writer registration.
"""

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    StepResult,
    init_episode_tracking,
    track_episode_step,
)
from marl_battlegrounds.evaluation import recording_identity, run_writer
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, System
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.run_writer import RunWriter, configuration_identity
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    make_standard_team_deathmatch_config,
)


@pytest.fixture(scope="module")
def terminal_info() -> EpisodeInfo:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("priest",), team_b_roster=("priest",), max_steps=1
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    _, state = env.reset(jax.random.key(30))
    zero = jnp.zeros(10, jnp.int32)
    info = cast(
        EpisodeInfo,
        jax.jit(env.step)(jax.random.key(31), state, Action(zero, zero, zero))[4],
    )
    assert bool(info.completed)
    return info


def _empty_starts() -> EpisodeStartRecords:
    return EpisodeStartRecords(
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.zeros(8, jnp.uint32),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
    )


def _details(writer: RunWriter) -> dict[str, Any]:
    return cast(
        dict[str, Any], json.loads((writer.run_dir / "run_details.json").read_text())
    )


def _reverse(value: jax.Array) -> jax.Array:
    return value[::-1]


def _repeat(value: jax.Array) -> jax.Array:
    return jnp.stack((value, value))


def _mutable(value: jax.Array) -> np.ndarray[Any, Any]:
    return np.array(value, copy=True)


@pytest.mark.parametrize("source_count", [1, 2, 3])
def test_ordered_bank_identity_preserves_existing_configuration_digest(
    terminal_info: EpisodeInfo, source_count: int
) -> None:
    sources = [
        terminal_info.config._replace(max_steps=jnp.int32(index + 1))
        for index in range(source_count)
    ]
    bank = (
        sources[0]
        if source_count == 1
        else jax.tree.map(lambda *values: jnp.stack(values), *sources)
    )
    identifier, references, contents = ordered_source_bank_identity(bank)
    expected = [configuration_identity(source) for source in sources]
    expected_refs = [value[0] for value in expected]
    payload = json.dumps(expected_refs, separators=(",", ":")).encode() + b"\n"
    assert references == expected_refs
    assert identifier == sha256(payload).hexdigest()
    assert contents == dict(expected)
    if source_count > 1:
        reversed_id = ordered_source_bank_identity(jax.tree.map(_reverse, bank))[0]
        assert reversed_id != identifier


def test_equal_source_rows_remain_distinct_ordered_references(
    terminal_info: EpisodeInfo,
) -> None:
    source = terminal_info.config
    bank = jax.tree.map(_repeat, source)
    identifier, references, contents = ordered_source_bank_identity(bank)
    assert references == [configuration_identity(source)[0]] * 2
    assert len(contents) == 1
    assert identifier != ordered_source_bank_identity(source)[0]


def test_empty_unchanged_starts_do_not_flush_pending_completion(
    tmp_path: Path, terminal_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = terminal_info.config
    starts = _empty_starts()

    @jax.jit
    def return_source(config: EnvConfig) -> EnvConfig:
        return config

    with RunWriter(tmp_path, buffer_size=8) as writer:
        writer.register_episodes(starts, source_configs=source)
        writer.write(terminal_info)
        manifest = (writer.run_dir / "run_details.json").read_bytes()

        def fail(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("empty cached registration performed output work")

        with monkeypatch.context() as guarded:
            guarded.setattr(writer, "flush", fail)
            guarded.setattr(run_writer, "_json_bytes", fail)
            guarded.setattr(recording_identity, "ordered_source_bank_identity", fail)
            for _ in range(3):
                writer.register_episodes(starts, source_configs=source)
                source = cast(EnvConfig, return_source(source))
                writer.register_episodes(starts, source_configs=source)
                writer.register_episodes(starts)
        assert (writer.run_dir / "run_details.json").read_bytes() == manifest
        assert not (writer.run_dir / "episodes.csv").exists()
        writer.flush()
        with (writer.run_dir / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["episode_id"] == str(int(terminal_info.episode_id))


def test_empty_mutable_bank_is_rechecked_and_changed_content_is_saved(
    tmp_path: Path, terminal_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = jax.tree.map(_mutable, terminal_info.config)
    original = recording_identity.ordered_source_bank_identity
    calls: list[str] = []

    def counted(source: EnvConfig) -> tuple[str, list[str], dict[str, object]]:
        result = original(source)
        calls.append(result[0])
        return result

    monkeypatch.setattr(recording_identity, "ordered_source_bank_identity", counted)
    with RunWriter(tmp_path) as writer:
        starts = _empty_starts()
        writer.register_episodes(starts, source_configs=bank)
        before = (writer.run_dir / "run_details.json").read_bytes()
        writer.register_episodes(starts, source_configs=bank)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        bank.max_steps[...] = 2
        writer.register_episodes(starts, source_configs=bank)
        assert len(calls) == 3 and calls[0] == calls[1] != calls[2]
        assert set(_details(writer)["source_banks"]) == set(calls)


@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_empty_starts_still_validate_every_field(
    tmp_path: Path, terminal_info: EpisodeInfo, fault: str
) -> None:
    starts = _empty_starts()
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(starts, source_configs=terminal_info.config)
        before = (writer.run_dir / "run_details.json").read_bytes()
        invalid = starts._replace(
            source_table_id=(
                jnp.zeros(9, jnp.uint32)
                if fault == "shape"
                else jnp.zeros(8, jnp.int32)
            )
        )
        with pytest.raises(ValueError, match="shape or dtype"):
            writer.register_episodes(invalid, source_configs=terminal_info.config)
        assert (writer.run_dir / "run_details.json").read_bytes() == before


def test_writer_excludes_final_before_transfer_and_flattening(
    tmp_path: Path, terminal_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = object()
    original = jax.device_get
    transfers: list[object] = []

    def checked(value: object) -> object:
        assert all(leaf is not final for leaf in jax.tree.leaves(value))
        transfers.append(value)
        return original(value)

    with RunWriter(tmp_path) as writer:
        monkeypatch.setattr(jax, "device_get", checked)
        writer.write(terminal_info._replace(final=cast(Any, final)))
        assert transfers
        writer.flush()
        assert (writer.run_dir / "episodes.csv").exists()


def test_lifecycle_error_rejects_before_final_or_record_transfer(
    tmp_path: Path, terminal_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    with RunWriter(tmp_path) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()

        def fail(_value: object) -> object:
            raise AssertionError("failed lifecycle reached payload transfer")

        monkeypatch.setattr(jax, "device_get", fail)
        with pytest.raises(ValueError, match="lifecycle_error"):
            writer.write(
                terminal_info._replace(
                    final=cast(Any, object()), lifecycle_error=jnp.asarray(True)
                )
            )
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not (writer.run_dir / "episodes.csv").exists()


def _never_call(*_args: object) -> object:
    raise AssertionError("recording executed the method")


def test_live_tracking_records_short_games_and_curriculum_ownership(
    tmp_path: Path, terminal_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(terminal_info.config._replace(max_steps=jnp.int32(n)) for n in (1, 3, 2)),
    )

    def select(indices: jax.Array) -> EnvConfig:
        def take(value: jax.Array) -> jax.Array:
            return value[indices]

        return balanced_spawn_configs(jax.tree.map(take, sources), num_envs=2)

    indices = jnp.asarray([0, 1], jnp.int32)
    env = marl_bgs.make("tdm", num_envs=2, env_config=select(indices), metrics="none")
    _, state = env.reset(jax.random.key(100))
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=sources,
        source_indices=indices,
        record_starts=True,
    ).begin_stage(state, total_env_steps=4)
    zero = jnp.zeros((2, 10), jnp.int32)
    actions = Action(zero, zero, zero)
    step = jax.jit(env.step)
    track = jax.jit(track_episode_step)
    system = System("fixed", _never_call, components=({"name": "fixed"},))
    expected: dict[int, tuple[int, int, int]] = {}
    completed: set[int] = set()
    hashes: list[str] = []
    original_identity = recording_identity.ordered_source_bank_identity

    def counted(source: EnvConfig) -> tuple[str, list[str], dict[str, object]]:
        result = original_identity(source)
        hashes.append(result[0])
        return result

    monkeypatch.setattr(recording_identity, "ordered_source_bank_identity", counted)
    fresh_bank_objects = False
    with RunWriter(
        tmp_path, phase="training", policies={"team_a": system, "team_b": system}
    ) as writer:
        for round_index in range(6):
            if round_index == 2:
                assert tracking.stage_summary(state)["status"] == "complete"
                tracking = tracking.begin_stage(state, total_env_steps=8)
                indices = jnp.asarray([2, 2], jnp.int32)
            _, state = env.reset_done(
                jax.random.key(200 + round_index), state, env_config=select(indices)
            )
            before = state
            result = cast(
                StepResult, step(jax.random.key(300 + round_index), state, actions)
            )
            previous_bank = tracking.source_configs
            tracking, result = cast(
                tuple[EpisodeTrackingState, StepResult],
                track(tracking, before, result, source_indices=indices),
            )
            fresh_bank_objects |= any(
                old is not new
                for old, new in zip(
                    jax.tree.leaves(previous_bank),
                    jax.tree.leaves(tracking.source_configs),
                    strict=True,
                )
            )
            _, state, _, _, info = result
            starts = info.episode_start_records
            assert starts is not None
            for lane in np.flatnonzero(np.asarray(starts.valid)):
                expected[int(starts.episode_id[lane])] = (
                    int(starts.source_index[lane]),
                    int(starts.spawn_locations[lane]),
                    int(starts.episode_start_stage[lane]),
                )
            completed.update(int(value) for value in info.episode_id[info.completed])
            writer.register_episodes(starts, source_configs=tracking.source_configs)
            trace = PolicyTrace(
                info.episode_id,
                info.decision_step,
                info.decision_step >= 0,
                jnp.where(info.active_mask, jnp.int32(0), jnp.int32(-1)),
            )
            writer.write(info, policy_trace=trace)
        assert tracking.stage_summary(state)["status"] == "complete"
        assert fresh_bank_objects
        assert len(hashes) == 1
        writer.flush()
        details = _details(writer)
        saved = next(iter(details["passes"].values()))
        assert len(expected) == 7
        assert expected[2] == (1, 1, 0)
        assert any(source == 2 and stage == 1 for source, _, stage in expected.values())
        assert set(saved["episode_starts"]) == {str(value) for value in expected}
        for episode_id, (source, spawn, stage) in expected.items():
            declaration = saved["episode_starts"][str(episode_id)]
            assert declaration["verification"] == "verified"
            assert declaration["source_index"] == source
            assert declaration["spawn_locations"] == spawn
            assert declaration["episode_start_stage"] == stage
        assert set(saved["completed_episode_ids"]) == completed
        with (writer.run_dir / "episodes.csv").open(newline="") as stream:
            outcomes = list(csv.DictReader(stream))
        assert {int(row["episode_id"]) for row in outcomes} == completed
        with (writer.run_dir / "policy_assignments.csv").open(newline="") as stream:
            assignments = list(csv.DictReader(stream))
        assert {int(row["episode_id"]) for row in assignments} == set(expected)


@pytest.mark.parametrize("change", ["value", "order", "signed_zero"])
def test_fresh_immutable_bank_checks_content_before_reusing_digest(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    source = terminal_info.config
    bank = jax.tree.map(_repeat, source)._replace(
        max_steps=jnp.array([1, 2], jnp.int32)
    )
    if change == "signed_zero":
        bank = bank._replace(obstacles=jnp.zeros_like(bank.obstacles))
    starts = _empty_starts()
    original = recording_identity.ordered_source_bank_identity
    hashes: list[str] = []

    def counted(config: EnvConfig) -> tuple[str, list[str], dict[str, object]]:
        result = original(config)
        hashes.append(result[0])
        return result

    monkeypatch.setattr(recording_identity, "ordered_source_bank_identity", counted)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(starts, source_configs=bank)

        def passthrough(config: EnvConfig) -> EnvConfig:
            return config

        fresh = cast(EnvConfig, jax.jit(passthrough)(bank))
        assert any(
            old is not new
            for old, new in zip(
                jax.tree.leaves(bank), jax.tree.leaves(fresh), strict=True
            )
        )
        writer.register_episodes(starts, source_configs=fresh)
        assert len(hashes) == 1
        if change == "value":
            changed = fresh._replace(
                max_steps=jnp.asarray(fresh.max_steps).at[0].set(3)
            )
        elif change == "order":
            changed = jax.tree.map(_reverse, fresh)
        else:
            changed = fresh._replace(obstacles=-fresh.obstacles)
        writer.register_episodes(starts, source_configs=changed)
        assert len(hashes) == 2 and hashes[0] != hashes[1]
        assert set(_details(writer)["source_banks"]) == set(hashes)
        assert len(writer._source_cache) == 1  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("leading", [(), (2,), (3, 2)])
def test_legacy_nine_argument_starts_remain_absent_across_leading_axes(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    leading: tuple[int, ...],
) -> None:
    scalar = _empty_starts()
    assert scalar.source_class_ids is None

    def broadcast(value: jax.Array) -> jax.Array:
        return jnp.broadcast_to(value, (*leading, *value.shape))

    starts = jax.tree.map(broadcast, scalar)
    assert starts.valid.shape == leading
    assert starts.source_class_ids is None

    def identity(value: EpisodeStartRecords) -> EpisodeStartRecords:
        return value

    restored = cast(EpisodeStartRecords, jax.jit(identity)(starts))
    assert restored.source_class_ids is None
    assert jax.tree.structure(restored) == jax.tree.structure(starts)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(starts, source_configs=terminal_info.config)
        before = (writer.run_dir / "run_details.json").read_bytes()
        writer.register_episodes(restored)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not writer.has_pending_numerical_starts
        saved = next(iter(_details(writer)["passes"].values()))
        assert saved["episode_starts"] == {}
