"""Check terminal replay summaries and honest optional metric coverage.

A real one-transition game supplies replay and report facts. Deliberately changed
summaries must fail before completion publication, including when a replay was
saved earlier or recovered. Replay configurations must agree with earlier
schedule, start, trace and completion evidence. Partial priority prefixes and
contradictory outcome flags must not be accepted as complete optional reports.
"""

import json
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.models import CodeRevisionV2
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.run_writer import RunWriter, configuration_identity
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.fixture(scope="module")
def terminal_info() -> EpisodeInfo:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("priest",), team_b_roster=("priest",), max_steps=1
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="full", replay_episodes=(1,))
    _, state = env.reset(jax.random.key(13))
    zero = jnp.zeros(10, dtype=jnp.int32)
    result = cast(
        tuple[Any, ...],
        jax.jit(env.step)(jax.random.key(14), state, Action(zero, zero, zero)),
    )
    info = cast(EpisodeInfo, result[4])
    assert bool(info.completed) and int(info.outcome) == 3
    assert info.priority is not None and info.full is not None
    assert info.replay is not None
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


def _pass(path: Path) -> dict[str, Any]:
    details = json.loads((path / "run_details.json").read_text())
    return cast(dict[str, Any], next(iter(details["passes"].values())))


def _wrong_summary(info: EpisodeInfo, field: str) -> EpisodeInfo:
    if field == "outcome":
        return info._replace(outcome=jnp.int32(1))
    if field == "length":
        return info._replace(
            decision_step=info.decision_step + 1,
            episode_length=info.episode_length + 1,
        )
    return info._replace(team_scores=info.team_scores.at[0].add(1))


@pytest.mark.parametrize("field", ["outcome", "length", "score"])
@pytest.mark.parametrize("route", ["combined", "prior", "resume"])
def test_replay_summary_mismatch_never_publishes_completion(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    provenance: dict[str, object],
    field: str,
    route: str,
) -> None:
    info = terminal_info._replace(priority=None, full=None)
    bad = _wrong_summary(info, field)
    writer = RunWriter(tmp_path, details=provenance)
    path = writer.run_dir
    try:
        if route != "combined":
            writer.write_replay(cast(ReplayPackets, info.replay))
            bad = bad._replace(replay=None)
            if route == "resume":
                writer.close()
                writer = RunWriter(resume_from=path, details=provenance)
        before = (path / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match=r"summary differs.*replay"):
            writer.write(bad)
        assert (path / "run_details.json").read_bytes() == before
        assert not (path / "episodes.csv").exists()
        assert _pass(path)["completed_episode_ids"] == []
        if route == "combined":
            assert not list(path.glob("replays/*.json"))
    finally:
        writer.close()


@pytest.mark.parametrize("mode", ["priority", "full"])
def test_partial_priority_prefix_cannot_claim_report_coverage(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    mode: Literal["priority", "full"],
) -> None:
    metric = terminal_info.priority if mode == "priority" else terminal_info.full
    assert metric is not None
    missing = PRIORITY_METRIC_NAMES.index("team_a_return")
    metric = metric._replace(valid=metric.valid.at[missing].set(False))
    info = terminal_info._replace(
        replay=None,
        priority=metric if mode == "priority" else None,
        full=metric if mode == "full" else None,
    )
    with RunWriter(tmp_path) as writer:
        with pytest.raises(ValueError, match="priority"):
            writer.write(info)
        path = writer.run_dir
    assert not list(path.glob("*.csv"))
    assert _pass(path)["recorded_metrics_by_episode"] == {}


@pytest.mark.parametrize("resume", [False, True])
def test_completion_configuration_must_match_its_earlier_replay(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    provenance: dict[str, object],
    resume: bool,
) -> None:
    writer = RunWriter(tmp_path, details=provenance)
    path = writer.run_dir
    try:
        writer.write_replay(cast(ReplayPackets, terminal_info.replay))
        if resume:
            writer.close()
            writer = RunWriter(resume_from=path, details=provenance)
        bad = terminal_info._replace(
            config=terminal_info.config._replace(max_steps=2),
            replay=None,
            priority=None,
            full=None,
        )
        before = (path / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="config"):
            writer.write(bad)
        assert (path / "run_details.json").read_bytes() == before
        assert not (path / "episodes.csv").exists()
        assert _pass(path)["completed_episode_ids"] == []
    finally:
        writer.close()


@pytest.mark.parametrize("resume", [False, True])
def test_later_completed_replay_id_blocks_whole_standalone_batch(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    provenance: dict[str, object],
    resume: bool,
) -> None:
    writer = RunWriter(tmp_path, details=provenance)
    path = writer.run_dir
    try:
        writer.write(terminal_info._replace(replay=None, priority=None, full=None))
        writer.flush()
        if resume:
            writer.close()
            writer = RunWriter(resume_from=path, details=provenance)
        packet = cast(ReplayPackets, terminal_info.replay)
        new_packet = packet._replace(episode_id=packet.episode_id + 1)

        def concatenate(first: jax.Array, second: jax.Array) -> jax.Array:
            return jnp.concatenate((first, second))

        packets = jax.tree.map(concatenate, new_packet, packet)
        before = (path / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="already completed"):
            writer.write_replay(packets)
        assert (path / "run_details.json").read_bytes() == before
        assert not list(path.glob("replays/*.json"))
    finally:
        writer.close()


@pytest.mark.parametrize("binding", ["schedule", "start", "trace"])
def test_standalone_replay_must_match_registered_configuration_evidence(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    provenance: dict[str, object],
    binding: str,
) -> None:
    declared = terminal_info._replace(
        config=terminal_info.config._replace(max_steps=2),
        completed=jnp.asarray(False),
        outcome=jnp.int32(0),
        replay=None,
        priority=None,
        full=None,
    )
    with RunWriter(
        tmp_path,
        details=provenance,
        policies={"team_a": "a", "team_b": "b"},
    ) as writer:
        if binding == "schedule":
            writer.register_episodes(
                [
                    {
                        "episode_id": 1,
                        "configuration_digest": configuration_identity(declared.config)[
                            0
                        ],
                    }
                ]
            )
        elif binding == "start":
            start = EpisodeStartRecords(
                declared.episode_id,
                jnp.int32(0),
                jnp.zeros(8, jnp.uint32),
                jnp.int32(-1),
                jnp.int32(-1),
                jnp.int32(-1),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(True),
            )
            writer.register_episodes(start)
            writer.write(declared._replace(episode_start_records=start))
        else:
            trace = PolicyTrace(
                declared.episode_id,
                declared.decision_step,
                jnp.asarray(True),
                jnp.full(10, -1, jnp.int32),
            )
            writer.write(declared, policy_trace=trace)
        writer.flush()
        path = writer.run_dir
        before = (path / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="config"):
            writer.write_replay(cast(ReplayPackets, terminal_info.replay))
        assert (path / "run_details.json").read_bytes() == before
        assert not list(path.glob("replays/*.json"))
        assert _pass(path)["completed_episode_ids"] == []


@pytest.mark.parametrize("mode", ["priority", "full"])
@pytest.mark.parametrize("name", ["team_a_win", "team_a_draw", "team_b_loss"])
def test_outcome_flags_must_match_the_required_outcome(
    tmp_path: Path,
    terminal_info: EpisodeInfo,
    mode: Literal["priority", "full"],
    name: str,
) -> None:
    metric = terminal_info.priority if mode == "priority" else terminal_info.full
    assert metric is not None
    column = PRIORITY_METRIC_NAMES.index(name)
    metric = metric._replace(
        values=metric.values.at[column].set(1 - metric.values[column])
    )
    info = terminal_info._replace(
        replay=None,
        priority=metric if mode == "priority" else None,
        full=metric if mode == "full" else None,
    )
    with RunWriter(tmp_path) as writer:
        with pytest.raises(ValueError, match="measurement differs"):
            writer.write(info)
        path = writer.run_dir
    assert not list(path.glob("*.csv"))
    assert _pass(path)["recorded_metrics_by_episode"] == {}
