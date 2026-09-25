"""Join selected transition packets into complete replay artifacts on the host.

ReplayCollector checks episode IDs and transition order across chunks. It keeps
packets in memory by default or writes numeric temporary spools when requested.
Only a real terminal packet completes a replay. Callers consume each write
iterator and close the collector to release unfinished temporary files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import jax
import numpy as np

from marl_battlegrounds.evaluation.replay_capture import ReplayPackets

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV4
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4

type ContextFactory = Callable[
    [ReplayPackets], tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]
]


def validate_packet_epoch(
    episode_id: int,
    index: int,
    *,
    expected: int,
    initial: bool,
    has_transition: bool,
    completed: bool,
) -> None:
    """Check one real replay decision using the shared sequence rules.

    All arguments are host scalars already checked for their numerical dtype.
    ``expected`` is the next zero-based transition index; ``completed`` says the
    ID has already finished. Raises ValueError for invalid IDs, missing/duplicate
    steps, an incorrect first-row marker or a packet without a Core transition.
    This changes no collector state and does not inspect actions or game rules.
    """
    if episode_id <= 0 or completed:
        raise ValueError("replay packet has an invalid or completed episode ID")
    if index != expected or initial != (index == 0):
        raise ValueError(f"episode {episode_id} replay packets have a gap or duplicate")
    if not has_transition:
        raise ValueError("valid replay packet does not contain a real transition")


@dataclass
class _Episode:
    """Ordered host packets and an optional owned spool for one episode."""

    context: EvaluationEpisodeContextV4
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

    def checkpoint_streams(self, directory: Path) -> list[dict[str, object]]:
        """Copy unfinished numeric streams into a new checkpoint directory.

        Parameters
        ----------
        directory : Path
            Existing empty directory owned by the checkpoint publisher. This
            method creates one ``<episode_id>.npylog`` file per open episode.

        Returns
        -------
        list of dict
            JSON-ready episode IDs, original contexts/runtime, record counts,
            leaf layouts, file lengths and SHA256 digests. No learner state is
            included. The checkpoint owner adds configuration identities.

        Notes
        -----
        Host-only. Copies in 1 MiB blocks and restores each live stream's cursor.
        It neither closes streams nor publishes a checkpoint. Only disk-backed
        collectors support this path. File or validation failures propagate;
        the caller owns cleanup and marks its writer failed.
        """
        from marl_battlegrounds.evaluation.recording_checkpoint import (
            COPY_BLOCK_BYTES,
            packet_layout,
        )

        if self._closed:
            raise RuntimeError("replay collector is closed")
        records: list[dict[str, object]] = []
        for episode_id, episode in sorted(self._episodes.items()):
            stream = episode.stream
            if stream is None:
                raise ValueError(
                    "recording checkpoints require disk-backed replay streams"
                )
            stream.flush()
            position = stream.tell()
            digest = sha256()
            size = 0
            try:
                stream.seek(0)
                first = next(episode.records())
                layout = packet_layout(first)
                stream.seek(0)
                with (directory / f"{episode_id}.npylog").open("xb") as target:
                    while block := stream.read(COPY_BLOCK_BYTES):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                    target.flush()
                    import os

                    os.fsync(target.fileno())
            finally:
                stream.seek(position)
            records.append(
                {
                    "episode_id": episode_id,
                    "context_id": episode.context.identity.episode_id,
                    "count": episode.count,
                    "context": episode.context.model_dump(mode="json"),
                    "runtime": episode.runtime.model_dump(mode="json"),
                    "path": f"open_replays/{episode_id}.npylog",
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                    "layout": layout,
                }
            )
        return records

    def restore_stream(
        self,
        packet: ReplayPackets,
        *,
        context: EvaluationEpisodeContextV4,
        runtime: RuntimeProvenanceV1,
        stream: BinaryIO,
        count: int,
    ) -> None:
        """Adopt one already validated replay prefix without calling its factory.

        Parameters
        ----------
        packet : ReplayPackets
            The prefix's first scalar row. It supplies the episode ID and the
            installed tree layout for later rows.
        context : EvaluationEpisodeContextV4
            The episode context saved with the checkpoint, adopted as given.
            This current context records the game's Red Zone depth (0.0 when the
            rule is off).
        runtime : RuntimeProvenanceV1
            The runtime provenance saved with the checkpoint, adopted as given.
        stream : BinaryIO
            A caller-owned anonymous binary file at its append end, holding the
            prefix's rows.
        count : int
            The prefix's positive validated row count.

        Raises
        ------
        ValueError
            "cannot restore this replay episode into the collector" when the
            collector is closed or the episode ID is already open or completed;
            "an open replay prefix must contain a real transition" when count is
            below 1.

        Notes
        -----
        Host-only. This collector takes stream ownership on success: close
        releases it with the other unfinished spools. On failure the caller
        still owns stream. The checkpoint reader owns full content, context and
        order validation; this method checks only the conditions above.
        """
        episode_id = int(packet.episode_id)
        if (
            self._closed
            or episode_id in self._episodes
            or episode_id in self._completed
        ):
            raise ValueError("cannot restore this replay episode into the collector")
        if count < 1:
            raise ValueError("an open replay prefix must contain a real transition")
        self._episodes[episode_id] = _Episode(
            context, runtime, jax.tree.structure(packet), stream, count
        )

    def restore_completed(self, episode_ids: Iterable[int]) -> None:
        """Restore durable completed IDs before admitting later replay packets.

        IDs are positive integers from the validated pass. They must not name an
        unfinished restored stream. This changes only collector duplicate guards;
        it does not fabricate replay files for games that were never captured.
        """
        values = set(episode_ids)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("completed replay IDs must be positive integers")
        if values & self._episodes.keys():
            raise ValueError("an open replay also appears completed")
        self._completed.update(values)

    def preflight(
        self,
        packets: ReplayPackets,
        *,
        completed_episode_ids: Iterable[int] = (),
    ) -> ReplayPackets:
        """Check the complete incoming packet sequence without recording it.

        Parameters
        ----------
        packets : ReplayPackets
            Numerical packets with capacity axes and optional leading chunk
            axes. Every leaf must share the bool valid array's leading shape.
            Valid rows need positive int32 IDs, consecutive int32 indices,
            matching initial flags and real Core transitions.
        completed_episode_ids : Iterable[int], default ()
            Episode summaries the caller intends to record in the same call.
            A positive ID must not remain pending after these packets. IDs with
            no selected replay are allowed; this does not require every game
            to have a replay.

        Returns
        -------
        ReplayPackets
            The supplied tree transferred to host NumPy leaves. Pass this tree
            to write to avoid another device transfer. No input is changed.

        Raises
        ------
        RuntimeError
            This collector was closed.
        TypeError
            packets is not ReplayPackets or a completion ID is not an integer.
        ValueError
            A leading shape, dtype, ID, transition order, initial flag or real
            transition is invalid, or a submitted completion still lacks its
            selected replay's terminal packet.

        Notes
        -----
        Host-only. This does not call the context factory, open/write spools,
        build artifacts or change pending/completed state. It keeps only the
        incoming host packet and temporary per-episode counters. Context and
        full replay validation still belong to write; this is a numerical
        sequence and completion check, not proof of arbitrary context metadata.
        """
        if self._closed:
            raise RuntimeError("replay collector is closed")
        if not isinstance(cast(object, packets), ReplayPackets):
            raise TypeError("replay packets must be ReplayPackets")
        host = jax.device_get(packets)
        valid = np.asarray(host.valid)
        if valid.dtype != np.bool_:
            raise ValueError("replay valid flags must have boolean dtype")
        shape = valid.shape
        for leaf in jax.tree.leaves(host):
            if np.shape(leaf)[: len(shape)] != shape:
                raise ValueError("replay leaves must share the valid leading shape")
        fields = {
            "episode_id": (host.episode_id, np.int32),
            "transition_index": (host.transition_index, np.int32),
            "initial": (host.initial, np.bool_),
            "has_transition": (host.info.transition_facts.has_transition, np.bool_),
            "terminated": (host.done.terminated, np.bool_),
            "truncated": (host.done.truncated, np.bool_),
        }
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name, (value, dtype) in fields.items():
            array = np.asarray(value)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"replay {name} has the wrong shape or dtype")
            arrays[name] = array.reshape(-1)
        pending = {
            identifier: episode.count for identifier, episode in self._episodes.items()
        }
        finished: set[int] = set()
        for row in np.flatnonzero(valid):
            episode_id = int(arrays["episode_id"][row])
            index = int(arrays["transition_index"][row])
            expected = pending.get(episode_id, 0)
            validate_packet_epoch(
                episode_id,
                index,
                expected=expected,
                initial=bool(arrays["initial"][row]),
                has_transition=bool(arrays["has_transition"][row]),
                completed=episode_id in self._completed or episode_id in finished,
            )
            if bool(arrays["terminated"][row] or arrays["truncated"][row]):
                pending.pop(episode_id, None)
                finished.add(episode_id)
            else:
                pending[episode_id] = index + 1
        for episode_id in completed_episode_ids:
            if isinstance(episode_id, bool) or not isinstance(episode_id, Integral):
                raise TypeError("completed replay episode IDs must be integers")
            if episode_id <= 0:
                raise ValueError("completed replay episode IDs must be positive")
            if int(episode_id) in pending:
                raise ValueError(
                    f"episode {episode_id} completed before its replay packets"
                )
        return host

    def write(self, packets: ReplayPackets) -> Iterator[ReplayArtifactV4]:
        """Consume selected packets and yield each replay as soon as it completes.

        Parameters
        ----------
        packets : ReplayPackets
            ReplayPackets with capacity axes, optionally preceded by chunk
            axes. valid gives the leading axes; rows are consumed in flat order.

        Yields
        ------
        ReplayArtifactV4
            One ReplayArtifactV4 per completed selected episode. Consume the iterator
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

        The complete numerical sequence is checked before its first mutation or
        context-factory call. A later invalid row cannot follow an already yielded
        replay from this call. Context/artifact validation can still fail during
        collection; callers must treat that failure as a failed recording call.
        The call transfers device packets to the host and mutates this collector.
        It keeps unfinished episodes for later chunks. Completion closes that
        episode's spool; the returned artifact owns its materialized replay data.
        """
        from marl_battlegrounds.evaluation.replay_v4 import replay_from_packets

        host = self.preflight(packets)
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
            episode = self._episodes.get(episode_id)
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
                del artifact

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
