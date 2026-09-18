"""Check bounded drains against direct recording and reject whole invalid calls.

Real terminal and continuing records feed the same writer through both routes.
Tests cover sparse metric tables, source starts, policy choices, replay ownership,
evidence-only padding, bad later rows and errors without emitted records. The
writer must publish no dependent output from a rejected drain and must not copy
learner data or flush merely because an unchanged evidence row was inspected.
"""

import csv
import json
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.collection_types import (
    CollectedAssignments,
    CollectedBatch,
    CollectedBuffers,
    CollectedCompletions,
    CollectedCounts,
    CollectedStarts,
    CollectionErrors,
    ConfigEvidence,
)
from marl_battlegrounds.evaluation.models import CodeRevisionV2
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, System
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.run_writer import RunWriter, configuration_identity
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _never_call(*_args: object) -> object:
    raise AssertionError("recording must not choose an action")


def _systems() -> dict[str, object]:
    method = System("Collection", _never_call, components=({"name": "component"},))
    return {"team_a": method, "team_b": method}


@pytest.fixture(scope="module")
def terminal() -> EpisodeInfo:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("priest",), team_b_roster=("priest",), max_steps=1
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="full", replay_episodes=(1,))
    _, state = env.reset(jax.random.key(20))
    zero = jnp.zeros(10, jnp.int32)
    info = cast(
        EpisodeInfo,
        jax.jit(env.step)(jax.random.key(21), state, Action(zero, zero, zero))[4],
    )
    assert bool(info.completed) and info.full is not None and info.replay is not None
    return info


@pytest.fixture(scope="module")
def provenance() -> dict[str, object]:
    return {
        "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
            mode="json"
        ),
        "runtime_provenance": capture_runtime_provenance("0.0.0").model_dump(
            mode="json"
        ),
    }


def _trace(info: EpisodeInfo) -> PolicyTrace:
    return PolicyTrace(
        info.episode_id,
        info.decision_step,
        jnp.asarray(True),
        jnp.where(info.active_mask, jnp.int32(0), jnp.int32(-1)),
    )


def _start(info: EpisodeInfo) -> EpisodeStartRecords:
    identifier = ordered_source_bank_identity(info.config)[0]
    words = np.frombuffer(bytes.fromhex(identifier), ">u4").astype(np.uint32)
    return EpisodeStartRecords(
        info.episode_id,
        jnp.int32(0),
        jnp.asarray(words),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(-1),
        jnp.asarray(True),
        jnp.asarray(False),
        jnp.asarray(True),
    )


def _batch(info: EpisodeInfo, *, trace: PolicyTrace | None = None) -> CollectedBatch:
    host = jax.device_get(info)

    def one(value: object) -> jax.Array:
        return jnp.expand_dims(jnp.asarray(value), 0)

    evidence = ConfigEvidence(
        one(host.episode_id), one(host.decision_step), jax.tree.map(one, host.config)
    )
    completed = bool(host.completed)
    count = 1 if completed else 0
    priority = (
        jax.tree.map(one, host.priority)
        if completed and host.priority is not None and host.full is None
        else None
    )
    full = jax.tree.map(one, host.full) if completed and host.full is not None else None
    refs = jnp.asarray([0] if completed else [], jnp.int32)
    completions = CollectedCompletions(
        one(host.episode_id)[:count],
        one(host.outcome)[:count],
        one(host.decision_step)[:count],
        one(host.episode_length)[:count],
        one(host.team_scores)[:count],
        refs,
        refs if priority is not None else jnp.full(count, -1, jnp.int32),
        refs if full is not None else jnp.full(count, -1, jnp.int32),
    )
    starts = None
    if host.episode_start_records is not None:
        starts = CollectedStarts(
            jax.tree.map(one, host.episode_start_records),
            one(host.episode_id),
            one(host.decision_step),
            jnp.asarray([0], jnp.int32),
        )
    assignments = None
    if trace is not None:
        assignments = CollectedAssignments(
            jax.tree.map(one, jax.device_get(trace)),
            one(host.episode_id),
            one(host.decision_step),
            one(host.episode_length),
            jnp.asarray([0], jnp.int32),
        )
    replay = host.replay
    if replay is not None:
        valid = replay.valid

        def select(value: jax.Array) -> jax.Array:
            return value[valid]

        replay = jax.tree.map(select, replay)
    buffers = CollectedBuffers(
        evidence, starts, completions, priority, full, assignments, replay
    )
    counts = CollectedCounts(
        *(
            jnp.int32(value)
            for value in (
                1,
                int(starts is not None),
                count,
                int(priority is not None),
                int(full is not None),
                int(assignments is not None),
                0 if replay is None else len(replay.valid),
            )
        )
    )
    errors = CollectionErrors(
        one(host.lifecycle_error),
        jnp.zeros(1, jnp.int32)
        if host.episode_tracking_error is None
        else one(host.episode_tracking_error),
        jnp.zeros(1, jnp.int32),
        *(jnp.int32(-1) for _ in range(5)),
    )
    return CollectedBatch(buffers, counts, errors)


def _write(
    writer: RunWriter, batch: CollectedBatch, *, source_configs: EnvConfig | None = None
) -> None:
    writer._write_collected(  # pyright: ignore[reportPrivateUsage]
        batch, source_configs=source_configs
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {key: value for key, value in row.items() if key != "run_id"} for row in rows
    ]


@pytest.mark.parametrize("mode", ["none", "priority", "full"])
@pytest.mark.parametrize("with_replay", [False, True])
def test_compact_and_direct_records_agree(
    tmp_path: Path,
    terminal: EpisodeInfo,
    provenance: dict[str, object],
    mode: str,
    with_replay: bool,
) -> None:
    info = terminal._replace(
        priority=terminal.priority if mode != "none" else None,
        full=terminal.full if mode == "full" else None,
        replay=terminal.replay if with_replay else None,
        episode_start_records=_start(terminal),
    )
    paths: list[Path] = []
    for route in ("direct", "collected"):
        with RunWriter(
            tmp_path / route, policies=_systems(), details=provenance
        ) as writer:
            if route == "direct":
                writer.register_episodes(
                    info.episode_start_records, source_configs=info.config
                )
                assert writer.has_pending_numerical_starts
                writer.write(info, policy_trace=_trace(info))
            else:
                _write(
                    writer, _batch(info, trace=_trace(info)), source_configs=info.config
                )
            assert not writer.has_pending_numerical_starts
            paths.append(writer.run_dir)
    assert {path.name for path in paths[0].glob("*.csv")} == {
        path.name for path in paths[1].glob("*.csv")
    }
    for path in paths[0].glob("*.csv"):
        assert _rows(path) == _rows(paths[1] / path.name)
    manifests = [json.loads((path / "run_details.json").read_text()) for path in paths]
    passes = [next(iter(manifest["passes"].values())) for manifest in manifests]
    for field in (
        "episode_starts",
        "recorded_metrics_by_episode",
        "completed_episode_ids",
        "trace_epochs",
        "trace_config_ids",
    ):
        assert passes[0][field] == passes[1][field]
    assert len(passes[0]["replays"]) == len(passes[1]["replays"]) == int(with_replay)


@pytest.mark.parametrize(
    "failure",
    [
        "lifecycle",
        "tracking",
        "collector",
        "trace_epoch",
        "start_epoch",
        "reference",
        "metric_owner",
        "outcome",
        "evidence_epoch",
        "padding_evidence",
    ],
)
def test_bad_drain_has_no_dependent_publication(
    tmp_path: Path,
    terminal: EpisodeInfo,
    failure: str,
) -> None:
    info = terminal._replace(replay=None, episode_start_records=_start(terminal))
    batch = _batch(info, trace=_trace(info))
    if failure in ("lifecycle", "tracking", "collector"):
        name = {
            "lifecycle": "lifecycle_error",
            "tracking": "episode_tracking_error",
            "collector": "code",
        }[failure]
        dtype = np.bool_ if failure == "lifecycle" else np.int32
        batch = batch._replace(
            errors=batch.errors._replace(**{name: np.ones(1, dtype)})
        )
    elif failure in ("evidence_epoch", "padding_evidence"):
        epoch = 1 if failure == "evidence_epoch" else -1
        batch = batch._replace(
            buffers=batch.buffers._replace(
                evidence=batch.buffers.evidence._replace(
                    decision_step=jnp.asarray([epoch], jnp.int32)
                )
            )
        )
    elif failure == "trace_epoch":
        assert batch.buffers.assignments is not None
        family = batch.buffers.assignments
        batch = batch._replace(
            buffers=batch.buffers._replace(
                assignments=family._replace(
                    trace=family.trace._replace(decision_step=np.ones(1, np.int32))
                )
            )
        )
    elif failure == "start_epoch":
        assert batch.buffers.starts is not None
        batch = batch._replace(
            buffers=batch.buffers._replace(
                starts=batch.buffers.starts._replace(decision_step=np.ones(1, np.int32))
            )
        )
    else:
        name = {
            "reference": "config_index",
            "metric_owner": "full_index",
            "outcome": "outcome",
        }[failure]
        value = 2 if failure == "reference" else -1 if failure == "metric_owner" else 0
        batch = batch._replace(
            buffers=batch.buffers._replace(
                completions=batch.buffers.completions._replace(
                    **{name: np.asarray([value], np.int32)}
                )
            )
        )
    with RunWriter(tmp_path, policies=_systems()) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError):
            _write(writer, batch, source_configs=info.config)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not list(writer.run_dir.glob("*.csv"))


def test_evidence_only_padding_preserves_binding_without_flushing(
    tmp_path: Path,
    terminal: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = terminal._replace(replay=None, priority=None, full=None)
    padding = info._replace(completed=jnp.asarray(False), decision_step=jnp.int32(-1))
    with RunWriter(tmp_path) as writer:
        writer.write(info)
        writer.register_episodes([])
        path = writer.run_dir
        before = (path / "run_details.json").read_bytes()
        with monkeypatch.context() as patch:
            patch.setattr(
                writer, "flush", lambda: pytest.fail("evidence-only drain flushed")
            )
            _write(writer, _batch(padding))
        assert (path / "run_details.json").read_bytes() == before


@pytest.mark.parametrize("route", ["direct", "collected"])
@pytest.mark.parametrize("change", ["value", "signed_zero"])
def test_no_output_mutation_rejects_existing_schedule_binding(
    tmp_path: Path,
    terminal: EpisodeInfo,
    route: str,
    change: str,
) -> None:
    info = terminal._replace(
        replay=None,
        priority=None,
        full=None,
        completed=jnp.asarray(False),
        decision_step=jnp.int32(-1),
    )
    config = info.config
    if change == "value":
        changed = config._replace(max_steps=config.max_steps + 1)
    else:
        pads = config.team_spawn_pad_positions.at[0, 4, 0].set(0.0)
        config = config._replace(team_spawn_pad_positions=pads)
        changed = config._replace(team_spawn_pad_positions=pads.at[0, 4, 0].set(-0.0))
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(
            [{"episode_id": 1, "config_id": configuration_identity(config)[0]}]
        )
        before = (writer.run_dir / "run_details.json").read_bytes()
        bad = info._replace(config=changed)
        with pytest.raises(ValueError, match="config"):
            if route == "direct":
                writer.write(bad)
            else:
                _write(writer, _batch(bad))
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not list(writer.run_dir.glob("*.csv"))


def test_pending_start_count_survives_resume_and_clears_after_evidence(
    tmp_path: Path,
    terminal: EpisodeInfo,
) -> None:
    info = terminal._replace(
        replay=None, priority=None, full=None, episode_start_records=_start(terminal)
    )
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(info.episode_start_records, source_configs=info.config)
        assert writer.has_pending_numerical_starts
        path = writer.run_dir
    with RunWriter(resume_from=path) as writer:
        assert writer.has_pending_numerical_starts
        writer.write(info)
        assert not writer.has_pending_numerical_starts


def test_bad_later_decision_blocks_starts_metrics_and_earlier_choices(
    tmp_path: Path,
    terminal: EpisodeInfo,
) -> None:
    first = terminal._replace(replay=None, episode_start_records=_start(terminal))
    second = first._replace(episode_id=jnp.int32(2))
    second = second._replace(episode_start_records=_start(second))
    left = _batch(first, trace=_trace(first))
    right = _batch(second, trace=_trace(second))
    assert right.buffers.starts is not None and right.buffers.assignments is not None
    assignments = right.buffers.assignments
    assignments = assignments._replace(
        config_index=assignments.config_index + 1,
        trace=assignments.trace._replace(
            policy_ids=assignments.trace.policy_ids.at[0, 5].set(99)
        ),
    )
    right = right._replace(
        buffers=right.buffers._replace(
            starts=right.buffers.starts._replace(
                config_index=jnp.asarray([1], jnp.int32)
            ),
            assignments=assignments,
            completions=right.buffers.completions._replace(
                config_index=jnp.asarray([1], jnp.int32),
                full_index=jnp.asarray([1], jnp.int32),
            ),
        )
    )

    def concatenate(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.concatenate((a, b))

    def add(a: jax.Array, b: jax.Array) -> jax.Array:
        return a + b

    joined = left._replace(
        buffers=jax.tree.map(concatenate, left.buffers, right.buffers),
        counts=jax.tree.map(add, left.counts, right.counts),
    )
    with RunWriter(tmp_path, policies=_systems(), buffer_size=1) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="component table"):
            _write(writer, joined, source_configs=first.config)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not list(writer.run_dir.glob("*.csv"))


def test_error_only_drain_rejects_before_transferring_evidence(
    tmp_path: Path,
    terminal: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    padding = terminal._replace(
        completed=jnp.asarray(False),
        decision_step=jnp.int32(-1),
        replay=None,
        priority=None,
        full=None,
        lifecycle_error=jnp.asarray(True),
    )
    batch = _batch(padding)
    original = jax.device_get

    def guarded(value: object) -> object:
        assert value is not batch.buffers
        return original(value)

    with RunWriter(tmp_path) as writer:
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", guarded)
            with pytest.raises(ValueError, match="lifecycle_error"):
                _write(writer, batch)
        assert not list(writer.run_dir.glob("*.csv"))


def test_start_verification_reuses_only_exact_successful_config_relationships(
    tmp_path: Path,
    terminal: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.evaluation.run_writer as writer_module
    import marl_battlegrounds.tasks as tasks

    validations: list[object] = []
    relationships: list[object] = []
    validate = tasks._validate_config_choices  # pyright: ignore[reportPrivateUsage]
    relationship = writer_module._source_relationship_check()  # pyright: ignore[reportPrivateUsage]

    def checked(config: EnvConfig, **kwargs: object) -> None:
        validations.append(config)
        validate(
            config,
            batched=cast(bool, kwargs["batched"]),
            both_spawn_choices=cast(bool, kwargs["both_spawn_choices"]),
        )

    def compared(a: EnvConfig, b: EnvConfig) -> object:
        relationships.append(a)
        return relationship(a, b)

    monkeypatch.setattr(tasks, "_validate_config_choices", checked)
    monkeypatch.setattr(writer_module, "_source_relationship_check", lambda: compared)
    source = terminal.config
    swapped = source._replace(
        team_spawn_pad_positions=source.team_spawn_pad_positions[::-1]
    )
    negative_zero = source._replace(obstacles=source.obstacles.at[-1, 2].set(-0.0))
    assert configuration_identity(source)[0] != configuration_identity(negative_zero)[0]
    with RunWriter(tmp_path, buffer_size=20) as writer:
        for episode, (config, choice) in enumerate(
            (
                (source, 0),
                (source, 0),
                (swapped, 1),
                (swapped, 1),
                (negative_zero, 0),
                (negative_zero, 0),
            ),
            start=1,
        ):
            info = terminal._replace(
                episode_id=jnp.int32(episode),
                config=config,
                replay=None,
                priority=None,
                full=None,
            )
            start = _start(terminal)._replace(
                episode_id=info.episode_id, spawn_locations=jnp.int32(choice)
            )
            info = info._replace(episode_start_records=start)
            _write(writer, _batch(info), source_configs=source)
        assert len(validations) == 3
        assert len(relationships) == 3
        assert len(writer._validated_start_configs) == 3  # pyright: ignore[reportPrivateUsage]
        assert len(writer._verified_source_choices) == 3  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("change", ["dtype", "shape", "wrong_choice"])
def test_verified_cache_cannot_admit_changed_types_shapes_or_declarations(
    tmp_path: Path,
    terminal: EpisodeInfo,
    change: str,
) -> None:
    first = terminal._replace(replay=None, priority=None, full=None)
    first = first._replace(episode_start_records=_start(first))
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(
            first.episode_start_records, source_configs=first.config
        )
        writer.write(first)
        writer.flush()
        host = cast(EpisodeInfo, jax.device_get(first))
        config = host.config
        start = _start(first)._replace(episode_id=np.int32(2))
        if change == "dtype":
            config = config._replace(
                team_spawn_pad_positions=np.asarray(
                    config.team_spawn_pad_positions
                ).astype(np.float64)
            )
            assert (
                configuration_identity(config)[0]
                == configuration_identity(first.config)[0]
            )
        elif change == "shape":
            config = config._replace(
                team_spawn_pad_positions=np.asarray(
                    config.team_spawn_pad_positions
                ).reshape((10, 2))
            )
        else:
            start = start._replace(spawn_locations=np.int32(1))
        bad = host._replace(
            episode_id=np.int32(2), config=config, episode_start_records=start
        )
        writer.register_episodes(start, source_configs=first.config)
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises((TypeError, ValueError)):
            writer.write(bad)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert writer.completed_episode_ids == frozenset({1})


def test_successful_verification_caches_are_bounded(
    tmp_path: Path,
    terminal: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.evaluation.run_writer as writer_module

    monkeypatch.setattr(writer_module, "_VERIFICATION_CACHE_SIZE", 2)
    with RunWriter(tmp_path) as writer:
        for identifier in range(1, 5):
            info = terminal._replace(
                episode_id=jnp.int32(identifier),
                config=terminal.config._replace(max_steps=jnp.int32(identifier)),
                replay=None,
                priority=None,
                full=None,
            )
            info = info._replace(episode_start_records=_start(info))
            _write(writer, _batch(info), source_configs=info.config)
            assert len(writer._validated_start_configs) <= 2  # pyright: ignore[reportPrivateUsage]
            assert len(writer._verified_source_choices) <= 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("route", ["direct", "collected"])
@pytest.mark.parametrize("resume", [False, True])
def test_completion_only_config_binding_survives_padding_and_resume(
    tmp_path: Path,
    terminal: EpisodeInfo,
    route: str,
    resume: bool,
) -> None:
    info = terminal._replace(
        priority=None, full=None, replay=None, episode_start_records=None
    )
    writer = RunWriter(tmp_path)
    writer.write(info)
    writer.flush()
    path = writer.run_dir
    if resume:
        writer.close()
        writer = RunWriter(resume_from=path)
    try:
        before = (path / "run_details.json").read_bytes()
        bad = info._replace(
            completed=jnp.asarray(False),
            outcome=jnp.int32(0),
            decision_step=jnp.int32(-1),
            config=info.config._replace(max_steps=info.config.max_steps + 1),
        )
        with pytest.raises(ValueError, match="configuration differs"):
            if route == "direct":
                writer.write(bad)
            else:
                _write(writer, _batch(bad))
        assert (path / "run_details.json").read_bytes() == before
    finally:
        writer.close()


def test_replay_preparation_has_no_live_spool_or_metadata_effects(
    tmp_path: Path,
    terminal: EpisodeInfo,
    monkeypatch: pytest.MonkeyPatch,
    provenance: dict[str, object],
) -> None:
    import marl_battlegrounds.evaluation.recording_context as context_module

    calls: list[int] = []

    def capture() -> dict[str, object]:
        calls.append(1)
        return provenance

    monkeypatch.setattr(context_module, "capture_recording_provenance", capture)
    assert terminal.replay is not None
    with RunWriter(tmp_path, policies=_systems()) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        prepared = writer._preflight_replays(terminal.replay, {}, {})  # pyright: ignore[reportPrivateUsage]
        assert prepared is not None and prepared.provenance == provenance
        assert calls == [1]
        assert writer._collector is None  # pyright: ignore[reportPrivateUsage]
        assert writer._preflight_contexts == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._replay_config_ids == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._replay_ids == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._recording_provenance is None  # pyright: ignore[reportPrivateUsage]
        assert "recording_provenance" not in writer._details["passes"][writer._pass_key]  # pyright: ignore[reportPrivateUsage]
        assert not (writer.run_dir / ".replay_spool").exists()
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        writer.write(terminal)
        assert (writer.run_dir / "replays").is_dir()


def test_bad_later_replay_identity_leaves_no_preflight_metadata(
    tmp_path: Path,
    terminal: EpisodeInfo,
    provenance: dict[str, object],
) -> None:
    assert terminal.replay is not None
    first = terminal.replay
    second = first._replace(episode_id=jnp.full_like(first.episode_id, 2))

    def concatenate(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.concatenate((a, b))

    packets = jax.tree.map(concatenate, first, second)
    with RunWriter(tmp_path, policies=_systems(), details=provenance) as writer:
        writer.register_episodes([{"episode_id": 2, "config_id": "wrong"}])
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="configuration differs"):
            writer.write_replay(packets)
        assert writer._collector is None  # pyright: ignore[reportPrivateUsage]
        assert writer._preflight_contexts == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._replay_config_ids == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._replay_ids == {}  # pyright: ignore[reportPrivateUsage]
        assert writer._recording_provenance is None  # pyright: ignore[reportPrivateUsage]
        assert not (writer.run_dir / ".replay_spool").exists()
        assert not (writer.run_dir / "replays").exists()
        assert (writer.run_dir / "run_details.json").read_bytes() == before


@pytest.mark.parametrize("route", ["direct", "collected"])
@pytest.mark.parametrize("invalid", ["nonfinite", "task_reward"])
def test_bad_later_replay_artifact_publishes_no_part_of_incoming_call(
    tmp_path: Path,
    terminal: EpisodeInfo,
    provenance: dict[str, object],
    route: str,
    invalid: str,
) -> None:
    first = terminal._replace(priority=None, full=None)
    assert first.replay is not None
    second_packets = first.replay._replace(
        episode_id=jnp.full_like(first.replay.episode_id, 2),
        reward=first.replay.reward._replace(
            rewards=first.replay.reward.rewards.at[0, 0].set(
                jnp.nan if invalid == "nonfinite" else 1.0
            )
        ),
    )
    second = first._replace(episode_id=jnp.int32(2), replay=second_packets)

    def concatenate(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.concatenate((a, b))

    def stack(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.stack((a, b))

    def add(a: jax.Array, b: jax.Array) -> jax.Array:
        return a + b

    with RunWriter(tmp_path, policies=_systems(), details=provenance) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError):
            if route == "direct":
                info = jax.tree.map(
                    stack,
                    first._replace(replay=None),
                    second._replace(replay=None),
                )
                writer.write(
                    info._replace(
                        replay=jax.tree.map(concatenate, first.replay, second_packets)
                    )
                )
            else:
                left, right = _batch(first), _batch(second)
                right = right._replace(
                    buffers=right.buffers._replace(
                        completions=right.buffers.completions._replace(
                            config_index=jnp.asarray([1], jnp.int32)
                        )
                    )
                )
                joined = left._replace(
                    buffers=jax.tree.map(concatenate, left.buffers, right.buffers),
                    counts=jax.tree.map(add, left.counts, right.counts),
                )
                _write(writer, joined)
        assert not list((writer.run_dir / "replays").glob("*.json"))
        assert not list(writer.run_dir.glob(".replay_publish-*"))
        assert not list(writer.run_dir.glob("*.csv"))
        assert writer._pending_replays == {}  # pyright: ignore[reportPrivateUsage]
        assert (writer.run_dir / "run_details.json").read_bytes() == before


def test_replay_staging_keeps_one_model_and_serializes_each_completion_once(
    tmp_path: Path,
    terminal: EpisodeInfo,
    provenance: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref
    from collections.abc import Iterable

    import marl_battlegrounds.evaluation.replay_io as replay_io
    import marl_battlegrounds.evaluation.replay_v3 as replay_v3
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV3
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_capture import ReplayPackets

    assert terminal.replay is not None
    references: list[weakref.ReferenceType[object]] = []
    publications: list[Path] = []
    build = replay_v3.replay_from_packets
    publish = replay_io.publish_prepared_replay

    def checked_build(
        context: EvaluationEpisodeContextV3,
        packets: Iterable[ReplayPackets],
        *,
        runtime_provenance: RuntimeProvenanceV1,
    ) -> replay_v3.ReplayArtifactV3:
        assert all(reference() is None for reference in references)
        replay = build(context, packets, runtime_provenance=runtime_provenance)
        references.append(weakref.ref(replay))
        return replay

    def staged_publish(
        prepared: replay_io.PreparedReplay,
        destination: replay_io.ReplayDestination,
        *,
        verify_existing_replay: bool = False,
    ) -> replay_io.SavedReplay:
        assert destination.replay_path.parent.name.startswith(".replay_publish-")
        assert not list((writer.run_dir / "replays").glob("*.json"))
        publications.append(destination.replay_path)
        return publish(
            prepared, destination, verify_existing_replay=verify_existing_replay
        )

    monkeypatch.setattr(replay_v3, "replay_from_packets", checked_build)
    monkeypatch.setattr(replay_io, "publish_prepared_replay", staged_publish)
    packets = jax.tree.map(
        lambda *values: jnp.concatenate(values),
        *(
            terminal.replay._replace(
                episode_id=jnp.full_like(terminal.replay.episode_id, identifier)
            )
            for identifier in range(1, 4)
        ),
    )
    with RunWriter(tmp_path, policies=_systems(), details=provenance) as writer:
        writer.write_replay(packets)
        assert len(references) == len(publications) == 3
        assert all(reference() is None for reference in references)
        assert len(list((writer.run_dir / "replays").glob("*.json"))) == 3
        assert not list(writer.run_dir.glob(".replay_publish-*"))


def test_staged_replay_conflicts_are_checked_before_any_final_link(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.replay_io import (
        ReplaySaveError,
        _publish_staged_replays,  # pyright: ignore[reportPrivateUsage]
    )

    source = tmp_path / "private"
    target = tmp_path / "replays"
    source.mkdir()
    target.mkdir()
    first, second = source / "first", source / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    final_first = target / "first.marlbg-replay.json"
    final_second = target / "second.marlbg-replay.json"
    final_second.write_bytes(b"wrong!")
    files = [(first, final_first, 5), (second, final_second, 6)]
    with pytest.raises(ReplaySaveError, match="differs"):
        _publish_staged_replays(files)
    assert not final_first.exists()
    assert final_second.read_bytes() == b"wrong!"
    final_second.write_bytes(b"second")
    _publish_staged_replays(files)
    assert final_first.read_bytes() == b"first"
    assert final_first.stat().st_ino == first.stat().st_ino
    _publish_staged_replays(files)
    assert final_second.read_bytes() == b"second"
