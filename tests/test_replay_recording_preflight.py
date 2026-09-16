"""Check replay sequence preflight before collection or dependent publication.

Invalid later rows and incomplete selected games must fail without calling the
context factory or changing pending spools. Host packets returned by preflight
remain usable by the ordinary collector, and unused capacity does no recording.
"""

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, DoneFlags
from marl_battlegrounds.evaluation.models import (
    CodeRevisionV2,
    EvaluationEpisodeContextV3,
)
from marl_battlegrounds.evaluation.recording_context import build_recording_context
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.fixture(scope="module")
def terminal() -> ReplayPackets:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("priest",), team_b_roster=("priest",), max_steps=1
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none", replay_episodes=(1,))
    _, state = env.reset(jax.random.key(1))
    zero = jnp.zeros(10, dtype=jnp.int32)
    result = cast(
        tuple[Any, ...],
        jax.jit(env.step)(jax.random.key(2), state, Action(zero, zero, zero)),
    )
    packet = cast(ReplayPackets, result[4].replay)
    assert packet.valid.shape == (1,) and bool(packet.done.done[0])
    return packet


def _context(
    packet: ReplayPackets,
) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
    return build_recording_context(
        packet.config,
        run_id="test",
        phase="evaluation",
        pass_id="1",
        episode={"episode_id": int(packet.episode_id)},
        policies={"team_a": "test", "team_b": "test"},
        details={
            "code_revision": CodeRevisionV2(package_version="0.0.0"),
            "runtime_provenance": capture_runtime_provenance("0.0.0"),
        },
    )


def _pair(left: ReplayPackets, right: ReplayPackets) -> ReplayPackets:
    def concatenate(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.concatenate((a, b))

    return jax.tree.map(concatenate, left, right)


@pytest.mark.parametrize("bad", ["gap", "initial", "transition", "completed", "dtype"])
def test_later_bad_row_never_calls_factory_or_writes_spools(
    tmp_path: Path, terminal: ReplayPackets, bad: str
) -> None:
    later = terminal._replace(episode_id=jnp.array([2], jnp.int32))
    if bad == "gap":
        later = later._replace(transition_index=jnp.array([1], jnp.int32))
    elif bad == "initial":
        later = later._replace(initial=jnp.array([False]))
    elif bad == "transition":
        later = later._replace(
            info=later.info._replace(
                transition_facts=later.info.transition_facts._replace(
                    has_transition=jnp.array([False])
                )
            )
        )
    elif bad == "completed":
        later = terminal
    else:
        later = later._replace(episode_id=jnp.array([2.0], jnp.float32))
    packets = _pair(terminal, later)
    calls: list[int] = []

    def context(
        packet: ReplayPackets,
    ) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
        calls.append(int(packet.episode_id))
        return _context(packet)

    collector = ReplayCollector(context, spool_dir=tmp_path)
    try:
        with pytest.raises(ValueError):
            collector.preflight(packets)
        with pytest.raises(ValueError):
            list(collector.write(packets))
        assert calls == [] and not collector.pending_episode_ids
        assert list(tmp_path.iterdir()) == []
    finally:
        collector.close()


def test_pending_completion_blocks_another_terminal_before_any_mutation(
    tmp_path: Path, terminal: ReplayPackets
) -> None:
    calls: list[int] = []

    def context(
        packet: ReplayPackets,
    ) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
        calls.append(int(packet.episode_id))
        return _context(packet)

    collector = ReplayCollector(context, spool_dir=tmp_path)
    pending = terminal._replace(done=DoneFlags(jnp.array([False]), jnp.array([False])))
    try:
        assert list(collector.write(pending)) == []
        episode = collector._episodes[1]  # pyright: ignore[reportPrivateUsage]
        assert episode.stream is not None
        before = (episode.count, episode.stream.tell())
        other_terminal = terminal._replace(episode_id=jnp.array([2], jnp.int32))
        with pytest.raises(ValueError, match="completed before its replay"):
            collector.preflight(other_terminal, completed_episode_ids=(1, 2))
        assert collector.pending_episode_ids == frozenset({1})
        assert (episode.count, episode.stream.tell()) == before
        assert calls == [1]
    finally:
        collector.close()


def test_preflight_returned_host_packet_records_and_unknown_completion_needs_no_replay(
    terminal: ReplayPackets,
) -> None:
    collector = ReplayCollector(_context)
    try:
        host = collector.preflight(terminal, completed_episode_ids=(1, 9))
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(host))
        assert not collector.pending_episode_ids
        artifacts = list(collector.write(host))
        assert len(artifacts) == 1
        with pytest.raises(ValueError, match="completed episode ID"):
            collector.preflight(terminal)
    finally:
        collector.close()


def test_padding_performs_no_recording_and_closed_preflight_fails(
    terminal: ReplayPackets,
) -> None:
    calls: list[object] = []

    def never_context(
        packet: ReplayPackets,
    ) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
        calls.append(packet)
        raise AssertionError("padding called context factory")

    collector = ReplayCollector(never_context)
    padding = terminal._replace(valid=jnp.array([False]))
    assert (
        list(collector.write(collector.preflight(padding, completed_episode_ids=(7,))))
        == []
    )
    assert calls == [] and not collector.pending_episode_ids
    collector.close()
    with pytest.raises(RuntimeError, match="closed"):
        collector.preflight(padding)
