"""Own one bounded DevClient worker and declared System installations.

The service submits setup or one joint decision, then explicitly accepts or
abandons its result. Each installation keeps its own context and resource stack.
Only the worker enters/exits those contexts. Model-call files stay beside the
recorder's destination; they never decide whether replay data is safely saved.
No worker, factory or connection starts until the service requests work.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from contextvars import Context
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Literal
from uuid import uuid4

import jax

from marl_battlegrounds._method_loading import load_factory
from marl_battlegrounds.evaluation.host_evidence import HostRun
from marl_battlegrounds.evaluation.policy_execution import Policy, System, shared_policy
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from scripts.dev.visual_debugger.input import InputDispatchResult
from scripts.dev.visual_debugger.model import DebuggerSession, is_system_controller


class AbandonedSystemDecisionError(RuntimeError):
    """The service did not adopt this prepared joint decision."""


@dataclass(eq=False)
class SystemInstallation:
    """Keep one owned setup's resources and evidence on its worker context.

    loaded contains actual factory returns for alias and singleton checks.
    session is the prepared initial snapshot. cleanup and decision must only be
    used through context on the single worker. closed becomes true after cleanup
    succeeds. A failed cleanup remains owned and must not be hidden.
    """

    context: Context = field(default_factory=Context)
    cleanup: ExitStack = field(default_factory=ExitStack)
    loaded: dict[str, System | Policy] = field(
        default_factory=lambda: dict[str, System | Policy]()
    )
    session: DebuggerSession | None = None
    evidence: HostRun | None = None
    decision: AbstractContextManager[bool] | None = None
    directory: Path | None = None
    metadata: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    closed: bool = False
    close_error: BaseException | None = None
    cancellation: Event = field(default_factory=Event)

    def check_cancelled(self) -> None:
        """Avoid starting another callback after the service has cancelled setup."""
        if self.cancellation.is_set():
            raise AbandonedSystemDecisionError("System setup was cancelled")

    def close(self) -> None:
        """Close this installation on its context after any current call settles."""
        if self.closed:
            return
        if self.close_error is not None:
            raise RuntimeError(
                "System cleanup failed; ownership is still retained"
            ) from self.close_error
        errors: list[BaseException] = []
        for action in (lambda: self.settle(False), self.cleanup.close):
            try:
                action()
            except BaseException as error:
                errors.append(error)
        if errors:
            self.close_error = BaseExceptionGroup("System cleanup failed", errors)
            raise self.close_error
        self.closed = True

    def settle(self, played: bool) -> None:
        """Finish the open decision once, then flush its separate call evidence.

        played comes only from service adoption, including a later recording
        error after an accepted step. False abandons the answer and its history.
        This does not mark a game or replay as saved.
        """
        scope, self.decision = self.decision, None
        if scope is not None:
            if played:
                scope.__exit__(None, None, None)
            else:
                error = AbandonedSystemDecisionError(
                    "Prepared decision was not accepted"
                )
                scope.__exit__(type(error), error, error.__traceback__)
        if self.evidence is not None:
            self.evidence.flush()

    def write_details(self) -> None:
        """Atomically save setup and accepted evidence, only when recording is on."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / "session.json"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.metadata, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


@dataclass(eq=False)
class PendingSystemOperation:
    """One setup or decision, retained until adoption or cancellation cleanup.

    base is the exact accepted snapshot used to start work. future finishes when
    a result is ready, not when it is played. cleanup_future keeps a cancelled
    slot occupied until late work and resource cleanup both finish.
    """

    kind: Literal["setup", "decision"]
    base: DebuggerSession
    installation: SystemInstallation
    future: Future[SystemInstallation] | Future[InputDispatchResult]
    identifier: str = field(default_factory=lambda: uuid4().hex)
    cancelled: bool = False
    adopted: bool = False
    cleanup_future: Future[None] | None = None
    acceptance_future: Future[None] | None = None
    replaced: SystemInstallation | None = None
    on_installed: Callable[[], None] | None = None


class SystemWorker:
    """Run declared factory setup and whole-System decisions on one owned thread.

    Constructing this helper starts nothing. The service may hold one active
    installation and one candidate. It must settle the current operation before
    submitting another. Cancellation cannot forcibly stop arbitrary Python code;
    its slot remains occupied until that code returns and cleanup finishes.
    """

    def __init__(self) -> None:
        """Start with no executor, installation or queued work."""
        self._executor: ThreadPoolExecutor | None = None
        self.operation: PendingSystemOperation | None = None
        self.active: SystemInstallation | None = None
        self._closed = False
        self.shutdown_futures: list[Future[None]] = []
        self.recording_futures: list[Future[None]] = []

    @staticmethod
    def _report_failure(future: Future[None]) -> None:
        """Log asynchronous recording/cleanup failures even after HTTP shutdown."""
        error = future.exception()
        if error is not None:
            logging.getLogger(__name__).error(
                "DevClient System cleanup or recording failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _pool(self) -> ThreadPoolExecutor:
        """Create the single worker on first selected use; reject use after close."""
        if self._closed:
            raise RuntimeError("System worker is closed")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="marl-devclient"
            )
        return self._executor

    def _idle(self) -> None:
        """Reject another setup or decision while the bounded slot is occupied."""
        if self.operation is not None:
            raise RuntimeError("A System operation is still pending")

    def prepare(
        self,
        base: DebuggerSession,
        declarations: Mapping[str, str],
        controllers: tuple[str, str],
        build: Callable[[tuple[System | None, System | None]], DebuggerSession],
        *,
        directory: Path | None,
    ) -> PendingSystemOperation:
        """Load only selected declared aliases and prepare a separate candidate.

        declarations maps bounded launcher aliases to trusted factory references.
        controllers follows Team A/B order. build constructs and initializes one
        replacement session using the loaded pair. directory is a recorder-owned
        adjacent evidence folder, or None to skip file work. The old installation
        remains intact until the service explicitly adopts this candidate.
        """
        self._idle()
        selected = {
            value.removeprefix("system:")
            for value in controllers
            if is_system_controller(value)
        }
        if not selected.issubset(declarations):
            raise ValueError("Choose a System declared when DevClient started")
        candidate = SystemInstallation(directory=directory)
        old = self.active

        def load() -> SystemInstallation:
            """Prepare one candidate entirely inside its own persistent context."""
            initial_resources = ExitStack()
            candidate.cleanup.enter_context(initial_resources)
            try:
                methods: dict[str, System] = {}
                for alias in sorted(selected):
                    candidate.check_cancelled()
                    loaded = load_factory(declarations[alias])
                    if old is not None and any(
                        loaded is value for value in old.loaded.values()
                    ):
                        raise ValueError(
                            "A factory must return a fresh System for replacement"
                        )
                    candidate.loaded[alias] = loaded
                    method = (
                        loaded if isinstance(loaded, System) else shared_policy(loaded)
                    )
                    methods[alias] = method
                    if method.resource_scope is not None:
                        initial_resources.enter_context(method.resource_scope(False))
                    candidate.check_cancelled()
                pair = tuple(
                    methods[value.removeprefix("system:")]
                    if is_system_controller(value)
                    else None
                    for value in controllers
                )
                candidate.check_cancelled()
                session = build((pair[0], pair[1]))
                candidate.check_cancelled()
                candidate.session = session
                identity = session.evaluation_context.identity
                candidate.metadata.update(
                    schema_version=1,
                    run_id=identity.run_id,
                    episode_id=identity.episode_id,
                    local_episode_id=1,
                    accepted_turns=0,
                    saved_prefixes=[],
                    systems={
                        str(team): normalize_system_registration(
                            method, phase="evaluation", frozen=True
                        )[1]
                        for team, method in enumerate(session.systems or ())
                    },
                    system_ids=session.system_ids,
                    attempts_are_not_saved_replays=True,
                )
                candidate.evidence = HostRun(
                    identity.run_id,
                    "devclient",
                    identity.episode_id,
                    directory,
                    candidate.metadata,
                )
                candidate.cleanup.enter_context(candidate.evidence.scope())
                candidate.cleanup.enter_context(initial_resources.pop_all())
                for method in methods.values():
                    if method.resource_scope is not None:
                        candidate.cleanup.enter_context(
                            method.resource_scope(directory is not None)
                        )
                candidate.write_details()
                return candidate
            except BaseException:
                candidate.close()
                raise

        future = self._pool().submit(candidate.context.run, load)
        operation = PendingSystemOperation("setup", base, candidate, future)
        self.operation = operation
        return operation

    def decide(
        self,
        base: DebuggerSession,
        dispatch: Callable[[], InputDispatchResult],
    ) -> PendingSystemOperation:
        """Prepare one complete joint step and keep its evidence outcome open.

        base must belong to the active installation. dispatch runs the existing
        input/control path once. Only the later service adoption settles its
        model answers as played. A thrown error abandons them on this context.
        """
        self._idle()
        installation = self.active
        if (
            installation is None
            or installation.closed
            or base.environment_state is None
        ):
            raise RuntimeError("No active System installation")

        def execute() -> InputDispatchResult:
            """Stage one result with evidence for the accepted decision epoch."""
            evidence = installation.evidence
            assert evidence is not None and base.environment_state is not None
            scope = evidence.decision(base.environment_state)
            installation.decision = scope
            scope.__enter__()
            try:
                result = dispatch()
                for leaf in jax.tree.leaves(result.session.state):
                    if isinstance(leaf, jax.Array):
                        leaf.block_until_ready()
                return result
            except BaseException:
                installation.settle(False)
                raise

        future = self._pool().submit(installation.context.run, execute)
        operation = PendingSystemOperation("decision", base, installation, future)
        self.operation = operation
        return operation

    def accept(
        self, operation: PendingSystemOperation, session: DebuggerSession
    ) -> None:
        """Queue played evidence and cleanup for an already installed result.

        The service calls this immediately at either real adoption site, before
        any later replay save or response step. Metadata/cleanup errors propagate
        through finish_accept without rewriting the accepted action as abandoned.
        """
        if self.operation is not operation or operation.cancelled:
            raise RuntimeError("System operation is no longer current")
        operation.adopted = True
        installation = operation.installation
        old = self.active
        operation.replaced = old

        def accepted() -> None:
            """Flush accepted evidence and close a replaced installation in order."""
            if operation.kind == "decision":
                installation.context.run(installation.settle, True)
                installation.metadata["accepted_turns"] = (
                    session.current_evaluation_frame.frame_index
                )
                if (
                    installation.directory is not None
                    and session.system_memory is not None
                ):
                    trace = jax.device_get(session.system_memory.policy_trace)
                    installation.metadata["last_policy_trace"] = {
                        name: getattr(trace, name).tolist() for name in trace._fields
                    }
                installation.context.run(installation.write_details)
            if old is not None and old is not installation:
                old.context.run(old.close)

        self.active = installation
        operation.acceptance_future = self._pool().submit(accepted)

    def finish_accept(self, operation: PendingSystemOperation) -> bool:
        """Release an adopted slot after evidence and old cleanup finish.

        Never wait on arbitrary user cleanup in the service thread. Errors keep
        both installations owned and block new work instead of hiding lost cleanup.
        """
        future = operation.acceptance_future
        if future is None or not future.done():
            return False
        future.result()
        if self.operation is operation:
            self.operation = None
        return True

    def cancel(self, operation: PendingSystemOperation) -> None:
        """Fence a result now; queue its owned cleanup after late work returns.

        A cancelled setup leaves the old active installation usable. A cancelled
        decision closes its uncertain method instance. The caller keeps the last
        accepted game/replay prefix and must prepare fresh methods before play.
        """
        if self.operation is not operation or operation.adopted:
            return
        if operation.cancelled:
            return
        operation.cancelled = True
        operation.installation.cancellation.set()
        installation = operation.installation
        operation.cleanup_future = self._pool().submit(
            installation.context.run, installation.close
        )

    def finish_cancel(self, operation: PendingSystemOperation) -> bool:
        """Clear a cancelled slot only after cleanup succeeds; never wait here."""
        future = operation.cleanup_future
        if future is None or not future.done():
            return False
        future.result()
        if self.active is operation.installation:
            self.active = None
        if self.operation is operation:
            self.operation = None
        return True

    def record_saved_prefix(
        self,
        *,
        installation: SystemInstallation,
        replay_id: str,
        path: Path,
        transitions: int,
    ) -> None:
        """Record the recorder's verified saved prefix without claiming game completion.

        Only call after the recorder has verified its published bundle. Repeated
        identical acknowledgements do nothing. A new Save As path remains linked
        to the same adjacent call evidence; existing files are never moved.
        """
        if installation.directory is None:
            return

        def publish() -> None:
            """Flush calls, then add the verified replay identity atomically."""
            assert installation.evidence is not None
            if not installation.closed:
                installation.evidence.flush()
            prefixes = installation.metadata["saved_prefixes"]
            assert isinstance(prefixes, list)
            value = {
                "replay_id": replay_id,
                "path": str(path),
                "transitions": transitions,
            }
            if value not in prefixes:
                prefixes.append(value)
                installation.write_details()

        future = self._pool().submit(installation.context.run, publish)
        future.add_done_callback(self._report_failure)
        self.recording_futures.append(future)

    def close(self) -> None:
        """Fence pending work and queue all owned cleanup, without stopping a server.

        Do not promise to stop arbitrary factory/Python code: a running callback
        can delay cleanup and Python process exit. Ordinary HTTP work keeps its
        Client deadlines. No new work may be submitted after this call.
        """
        if self._closed:
            return
        if self.operation is not None:
            self.cancel(self.operation)
        active = self.active
        if active is not None and (
            self.operation is None
            or self.operation.adopted
            or active is not self.operation.installation
        ):
            self.shutdown_futures.append(
                self._pool().submit(active.context.run, active.close)
            )
        for future in self.shutdown_futures:
            future.add_done_callback(self._report_failure)
        if self.operation is not None:
            for future in (
                self.operation.cleanup_future,
                self.operation.acceptance_future,
            ):
                if future is not None:
                    future.add_done_callback(self._report_failure)
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False)
