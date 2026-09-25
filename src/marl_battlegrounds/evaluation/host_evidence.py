"""Bind optional host-method evidence to the runner's real game decisions.

The evaluator owns run identity, decision boundaries and durable game IDs. A
provider registers an observer within this host scope. Observers own their
separate call records. Nothing here enters actor inputs, numerical state or
compiled System execution. With no observers, decision scopes do no device work.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import jax
import numpy as np

from marl_battlegrounds.environment import EnvironmentState


@dataclass(frozen=True)
class DecisionContext:
    """Name pre-step lanes without adding game identifiers to actor inputs.

    episode_ids, steps, generations and valid have one entry per stable batch
    lane. steps counts decisions since that game's first state. generations
    fences lane replacements. A provider must not expose this envelope to its
    model or custom prompt/parser functions.
    """

    episode_ids: tuple[int, ...]
    steps: tuple[int, ...]
    generations: tuple[int, ...]
    valid: tuple[bool, ...]


class EvidenceObserver(Protocol):
    """Receive host decisions while keeping provider files outside game tables."""

    def decision(self, context: DecisionContext) -> AbstractContextManager[None]:
        """Begin a joint decision; normal exit means its step completed."""
        ...

    def flush(self) -> None:
        """Flush provider evidence before game publication can become durable."""
        ...

    def completed(self, episode_ids: frozenset[int]) -> None:
        """Observe only game IDs already durable under the existing writer."""
        ...

    def close(self) -> None:
        """Finish owned evidence files without changing game completion truth."""
        ...


_CURRENT: ContextVar[HostRun | None] = ContextVar("marl_host_evidence", default=None)
_TEAM: ContextVar[int | None] = ContextVar("marl_host_evidence_team", default=None)


@dataclass
class HostRun:
    """Own optional observers for one runner pass and its normal result metadata.

    run_id, phase and pass_id come from the runner. directory is the existing
    run directory, or None for an unsaved run. metadata is the result's existing
    host dictionary. Observers may add their own summaries to it. No observer
    owns the game writer or changes its completion boundary.
    """

    run_id: str
    phase: str
    pass_id: str
    directory: Path | None
    metadata: dict[str, object]
    publish: Callable[[str, str, dict[str, object]], None] | None = None
    previous: frozenset[int] = frozenset()
    observers: dict[str, EvidenceObserver] = field(
        default_factory=lambda: dict[str, EvidenceObserver]()
    )
    paths: dict[str, Path] = field(default_factory=lambda: dict[str, Path]())

    def register(
        self, name: str, factory: Callable[[], EvidenceObserver]
    ) -> EvidenceObserver:
        """Reuse one provider observer in this pass, creating it only when used."""
        if name not in self.observers:
            self.observers[name] = factory()
        return self.observers[name]

    @contextmanager
    def scope(self) -> Generator[None]:
        """Bind this run on its execution thread and close all observers on exit."""
        token = _CURRENT.set(self)
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                self.close()
            except BaseException as error:
                if body_error is None:
                    raise
                body_error.add_note(f"Host evidence cleanup also failed: {error}")
            finally:
                _CURRENT.reset(token)

    @contextmanager
    def decision(self, state: EnvironmentState) -> Generator[bool]:
        """Bind pre-step identities only when an observer needs host evidence.

        Yield True when the caller must wait for the numerical step to finish
        before normal context exit. Exceptions reach every observer. Empty scopes
        yield False without copying state or adding synchronization.
        """
        if not self.observers:
            yield False
            return
        ids, steps, generations, valid = jax.device_get(
            (
                state.episode_id,
                state.core_state.step_count - state.initial_step_count,
                state.reset_generation,
                ~state.done.done,
            )
        )
        context = DecisionContext(
            tuple(int(v) for v in np.asarray(ids).reshape(-1)),
            tuple(int(v) for v in np.asarray(steps).reshape(-1)),
            tuple(int(v) for v in np.asarray(generations).reshape(-1)),
            tuple(bool(v) for v in np.asarray(valid).reshape(-1)),
        )
        with ExitStack() as cleanup:
            for observer in self.observers.values():
                cleanup.enter_context(observer.decision(context))
            yield True

    def flush(self) -> None:
        """Flush provider evidence before the existing game writer publishes."""
        for observer in self.observers.values():
            observer.flush()

    def completed(self, episode_ids: frozenset[int]) -> None:
        """Pass through already durable IDs; never infer them from a transition."""
        for observer in self.observers.values():
            observer.completed(episode_ids)

    def close(self) -> None:
        """Attempt every observer cleanup and retain all failures."""
        errors: list[BaseException] = []
        for observer in self.observers.values():
            try:
                observer.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Host evidence cleanup failed", errors)


def current_run() -> HostRun | None:
    """Return this thread's runner evidence scope, or None for a direct loop."""
    return _CURRENT.get()


def current_team() -> int | None:
    """Return the fixed team label for evidence only, never model input."""
    return _TEAM.get()


@contextmanager
def team_scope(team: int) -> Generator[None]:
    """Bind one host call's team and restore its parent's binding on exit."""
    token = _TEAM.set(team)
    try:
        yield
    finally:
        _TEAM.reset(token)
