"""Join selected transition packets into complete replay artifacts on the host.

ReplayCollector checks episode IDs and transition order across chunks. It keeps
packets in memory by default or writes numeric temporary spools when requested.
Only a real terminal packet completes a replay. Callers consume each write
iterator and close the collector to release unfinished temporary files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import jax
import numpy as np

from marl_battlegrounds.evaluation.replay_capture import ReplayPackets

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV3
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3

type ContextFactory = Callable[
    [ReplayPackets], tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]
]


@dataclass
class _Episode:
    """Ordered host packets and an optional owned spool for one episode."""

    context: EvaluationEpisodeContextV3
    runtime: RuntimeProvenanceV1
    tree: Any  # JAX exposes PyTreeDef as a runtime alias, without a public type stub.
    stream: BinaryIO | None
    count: int = 0
    packets: list[ReplayPackets] = field(default_factory=lambda: list[ReplayPackets]())

    def append(self, packet: ReplayPackets) -> None:
        """Keep one packet in memory or write a pickle-free numeric spool row."""
        if self.stream is None:
            self.packets.append(packet)
        else:
            leaves = jax.tree.leaves(packet)
            # One self-describing numeric record; loading never enables pickle.
            dtype = np.dtype(
                [
                    (str(index), np.asarray(value).dtype, np.asarray(value).shape)
                    for index, value in enumerate(leaves)
                ]
            )
            record = np.empty((), dtype=dtype)
            for index, value in enumerate(leaves):
                record[str(index)] = np.asarray(value)
            np.save(self.stream, record, allow_pickle=False)
        self.count += 1

    def records(self) -> Iterator[ReplayPackets]:
        """Yield packets in order, rewinding any spool before reading."""
        if self.stream is None:
            yield from self.packets
        else:
            self.stream.seek(0)
            for _ in range(self.count):
                record = cast(
                    np.ndarray[Any, Any], np.load(self.stream, allow_pickle=False)
                )
                yield cast(
                    ReplayPackets,
                    self.tree.unflatten(
                        [record[str(index)] for index in range(self.tree.num_leaves)]
                    ),
                )

    def close(self) -> None:
        """Close the owned spool, if present."""
        if self.stream is not None:
            self.stream.close()


class ReplayCollector:
    """Keep only selected episodes; validate ordering before publishing each result.

    With ``spool_dir``, numeric packets use temporary files rather than an
    episode-length Python list. Final artifact construction still needs memory
    for that one replay. Consume ``write`` fully; it yields each finished replay
    before building the next, allowing a writer to persist it immediately.

    Parameters
    ----------
    context_factory : object
        Called with an episode's first valid scalar packet;
        returns its validated context and runtime provenance.
    spool_dir : object
        Optional temporary-file folder. None keeps packet lists in
        memory. A supplied directory is created if necessary.

    The collector owns any temporary files and must be closed after use. It
    tracks IDs for this recording scope; an already completed ID cannot restart.
    Host calls may transfer packet arrays from the device; no method is jittable.
    """

    def __init__(
        self, context_factory: ContextFactory, *, spool_dir: Path | None = None
    ) -> None:
        """Create an empty host collector and its optional spool directory."""
        self._factory = context_factory
        self._spool_dir = spool_dir
        self._episodes: dict[int, _Episode] = {}
        self._completed: set[int] = set()
        self._closed = False
        if spool_dir is not None:
            spool_dir.mkdir(parents=True, exist_ok=True)

    @property
    def pending_episode_ids(self) -> frozenset[int]:
        """Return the immutable set of episode IDs whose replay is still incomplete."""
        return frozenset(self._episodes)

    def write(self, packets: ReplayPackets) -> Iterator[ReplayArtifactV3]:
        """Consume selected packets and yield each replay as soon as it completes.

        Parameters
        ----------
        packets : ReplayPackets
            ReplayPackets with capacity axes, optionally preceded by chunk
            axes. valid gives the leading axes; rows are consumed in flat order.

        Yields
        ------
        ReplayArtifactV3
            One ReplayArtifactV3 per completed selected episode. Consume the iterator
            fully: transfers, validation and collection occur during iteration.

        Raises
        ------
        RuntimeError
            The collector was closed.
        ValueError
            A packet has an invalid/reused ID, a missing or repeated
            transition index, a wrong initial flag or no real transition.
        OSError
            A temporary spool cannot be read or written. Context/artifact
            validation errors propagate before that replay is yielded.

        The call transfers packets to the host and mutates this collector. It keeps
        unfinished episodes for later chunks. Completion closes that episode's spool;
        the returned artifact owns its materialized replay data.
        """
        from marl_battlegrounds.evaluation.replay_v3 import replay_from_packets

        if self._closed:
            raise RuntimeError("replay collector is closed")
        host = jax.device_get(packets)
        valid = np.asarray(host.valid)
        leading = valid.ndim

        def flatten(value: object) -> np.ndarray[Any, Any]:
            """Flatten packet-leading axes while retaining each payload shape."""
            array = np.asarray(value)
            return array.reshape((-1, *array.shape[leading:]))

        rows = jax.tree.map(flatten, host)
        for row in np.flatnonzero(valid):

            def select(value: object, index: int = int(row)) -> np.ndarray[Any, Any]:
                """Read one valid scalar packet from the flattened host batch."""
                return np.asarray(value)[index]

            packet = cast(ReplayPackets, jax.tree.map(select, rows))
            episode_id = int(packet.episode_id)
            index = int(packet.transition_index)
            if episode_id <= 0 or episode_id in self._completed:
                raise ValueError("replay packet has an invalid or completed episode ID")
            episode = self._episodes.get(episode_id)
            expected = 0 if episode is None else episode.count
            if index != expected or bool(packet.initial) != (index == 0):
                raise ValueError(
                    f"episode {episode_id} replay packets have a gap or duplicate"
                )
            if not bool(packet.info.transition_facts.has_transition):
                raise ValueError(
                    "valid replay packet does not contain a real transition"
                )
            if episode is None:
                context, runtime = self._factory(packet)
                episode = _Episode(
                    context,
                    runtime,
                    jax.tree.structure(packet),
                    None
                    if self._spool_dir is None
                    else TemporaryFile(dir=self._spool_dir),  # noqa: SIM115 - episode owns it
                )
                self._episodes[episode_id] = episode
            episode.append(packet)
            if bool(packet.done.done):
                try:
                    artifact = replay_from_packets(
                        episode.context,
                        episode.records(),
                        runtime_provenance=episode.runtime,
                    )
                finally:
                    episode.close()
                del self._episodes[episode_id]
                self._completed.add(episode_id)
                yield artifact

    def close(self) -> None:
        """Release every incomplete spool and mark the collector closed.

        Returns
        -------
        None
            None. Repeated calls are safe; later write iteration raises RuntimeError.

        Incomplete packets are discarded rather than published as completed replay
        evidence. Already returned artifacts remain usable. No final game step or
        automatic completion is invented.
        """
        for episode in self._episodes.values():
            episode.close()
        self._episodes.clear()
        self._closed = True
