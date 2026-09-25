"""Keep LLM call evidence separate from game records and actor memory.

The shared host runner supplies real game/decision boundaries. This module owns
bounded current-decision rows, per-team call totals and optional JSONL files.
RunWriter alone decides which games are durable. read_calls streams call rows;
full rows retain the exact serialized request, while light rows retain its hash.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO, cast
from uuid import uuid4

from marl_battlegrounds.evaluation.host_evidence import (
    DecisionContext,
    HostRun,
    current_run,
    current_team,
)
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.policies.actor import ActorAction

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.results import (
        EvaluationResult,
        SavedResults,
        TournamentResult,
    )

type RecordingMode = Literal["none", "light", "full"]
_CURRENT: ContextVar[_Decision | None] = ContextVar("marl_llm_decision", default=None)
_COUNTERS = (
    "model_calls",
    "tokenizer_calls",
    "input_tokens",
    "output_tokens",
    "missing_usage_replies",
    "reply_failures",
    "transport_failures",
    "fallback_actions",
    "forced_actions",
    "played_actions",
    "abandoned_actions",
    "unknown_actions",
)


class CallEvidence:
    """Hold one actor's prospective choice until its joint decision ends.

    The provider owns updates before sealing. Late updates are ignored after the
    decision closes, so a cancelled request cannot enter a replacement game.
    Even disabled recording keeps only small counters/action facts, never prompt
    hashes or copied request/reply text. No reference reaches a custom hook.
    """

    def __init__(
        self, decision: _Decision, row: dict[str, Any], mode: RecordingMode
    ) -> None:
        """Keep one small bounded row and its enclosing decision's lock."""
        self.decision = decision
        self.row = row
        self.mode = mode
        self.started = time.monotonic()

    def update(self, **values: object) -> None:
        """Update only this still-open decision, under its common lock."""
        with self.decision.lock:
            if not self.decision.closed:
                self.row.update(values)

    def count(self, name: str, value: int = 1) -> None:
        """Add an observed call/cost count while this attempt remains current."""
        with self.decision.lock:
            if not self.decision.closed:
                self.row[name] = self.row.get(name, 0) + value

    def request(self, request: Mapping[str, object]) -> None:
        """Count a generation attempt and optionally keep its exact wire payload."""
        self.count("model_calls")
        if self.mode == "none":
            return
        wire = json.dumps(dict(request), allow_nan=False, separators=(",", ":"))
        values: dict[str, object] = {
            "request_sha256": sha256(wire.encode()).hexdigest()
        }
        if self.mode == "full":
            values["request_json"] = wire
        self.update(**values)

    def response(self, response: Mapping[str, Any]) -> None:
        """Keep measured usage and the provider reply; never invent missing usage."""
        usage = response.get("usage")
        complete = True
        for source, target in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
        ):
            value: object = (
                cast(dict[str, object], usage).get(source)
                if isinstance(usage, dict)
                else None
            )
            if type(value) is int and value >= 0:
                self.count(target, value)
            else:
                complete = False
        if not complete:
            self.count("missing_usage_replies")
        if self.mode != "none":
            self.update(reply=response.get("choices"))

    def action(self, action: ActorAction, source: str, *, frame: str) -> None:
        """Keep the checked submitted choice; outcome is decided after the step."""
        self.update(
            action=[int(value) for value in action],
            source=source,
            frame=frame,
            seconds=time.monotonic() - self.started,
        )

    def failure(self, category: str, error: BaseException) -> None:
        """Record a declared failure class without copying arbitrary error text."""
        if category in ("reply_failures", "transport_failures"):
            self.count(category)
        self.update(failure=type(error).__name__)


class _Decision:
    """Keep only the current full joint decision's actor rows and fencing flag."""

    def __init__(self, owner: _Recorder, context: DecisionContext) -> None:
        """Bind the immutable runner identity envelope; expose none of it to actors."""
        self.owner = owner
        self.context = context
        self.lock = threading.Lock()
        self.closed = False
        self.rows: dict[tuple[int, int, int], CallEvidence] = {}
        for episode, valid in zip(context.episode_ids, context.valid, strict=True):
            if valid:
                owner.summary["episodes"].setdefault(
                    str(episode),
                    {
                        name: dict.fromkeys(_COUNTERS, 0)
                        for name in ("team_a", "team_b")
                    },
                )

    def actor(
        self, team: int, lane: int, actor: int, mode: RecordingMode, method_id: str
    ) -> CallEvidence:
        """Allocate one evidence channel for one valid actor in this decision."""
        with self.lock:
            if self.closed or not self.context.valid[lane]:
                raise RuntimeError(
                    "Cannot attach a call to a closed or inactive decision"
                )
            key = (team, lane, actor)
            if key in self.rows:
                raise RuntimeError("An actor was requested twice in one joint decision")
            enabled = mode if self.owner.run.directory is not None else "none"
            row: dict[str, Any] = {
                "episode_id": self.context.episode_ids[lane],
                "decision_step": self.context.steps[lane],
                "reset_generation": self.context.generations[lane],
                "team": team,
                "actor": actor,
                "method_id": method_id,
                "outcome": "unknown",
                "source": None,
                "action": None,
            }
            call = CallEvidence(self, row, enabled)
            self.rows[key] = call
            return call

    def finish(self, outcome: str, error: BaseException | None) -> None:
        """Seal late updates and publish known outcomes on the runner thread."""
        with self.lock:
            self.closed = True
            calls = tuple(self.rows.values())
            for call in calls:
                call.row["outcome"] = outcome
                call.row.setdefault("seconds", time.monotonic() - call.started)
                if error is not None:
                    call.row["abandoned_reason"] = type(error).__name__
        self.owner.publish(calls)


class _Recorder:
    """Own one execution attempt's streaming evidence and small summary counters."""

    def __init__(self, run: HostRun) -> None:
        """Create metadata only; defer directory/file creation until a saved row."""
        self.run = run
        self.attempt = uuid4().hex
        self.stream: TextIO | None = None
        self.sequence = 0
        self.acknowledged: set[int] = set(run.previous)
        self.directory_synced = False
        self.summary: dict[str, Any] = {
            "attempt_id": self.attempt,
            "teams": {
                name: dict.fromkeys(_COUNTERS, 0) for name in ("team_a", "team_b")
            },
            "methods": {},
            "episodes": {},
        }
        run.metadata["llm"] = self.summary

    def method(self, identity_json: str) -> str:
        """Register frozen method declarations using the existing digest owner."""
        identity = json.loads(identity_json)
        identifier = canonical_digest_sha256(identity)
        self.summary["methods"][identifier] = identity
        return identifier

    @contextmanager
    def decision(self, context: DecisionContext) -> Generator[None]:
        """Mark answers played only after the shared runner confirms its step."""
        decision = _Decision(self, context)
        token = _CURRENT.set(decision)
        try:
            yield
        except BaseException as error:
            decision.finish(
                "abandoned" if isinstance(error, Exception) else "unknown", error
            )
            raise
        else:
            decision.finish("played", None)
        finally:
            _CURRENT.reset(token)

    def _write(self, row: Mapping[str, object]) -> None:
        """Append one bounded JSON line; ordinary writer boundaries own durability."""
        if self.stream is None:
            assert self.run.directory is not None
            directory = self.run.directory / "model_calls"
            directory.mkdir(exist_ok=True)
            path = directory / f"{self.attempt}.jsonl"
            self.stream = path.open("x", encoding="utf-8")
            self.run.paths["model_calls"] = directory
            self.stream.write(
                json.dumps(
                    {
                        "type": "run",
                        "schema": "marl-llm-calls-v1",
                        "run_id": self.run.run_id,
                        "phase": self.run.phase,
                        "pass_id": self.run.pass_id,
                        "attempt_id": self.attempt,
                        "methods": self.summary["methods"],
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        self.stream.write(
            json.dumps(dict(row), allow_nan=False, separators=(",", ":")) + "\n"
        )

    def publish(self, calls: tuple[CallEvidence, ...]) -> None:
        """Update both teams' totals and write only requested actor evidence."""
        for call in calls:
            row = call.row
            team = "team_a" if row["team"] == 0 else "team_b"
            episode = self.summary["episodes"].setdefault(
                str(row["episode_id"]),
                {name: dict.fromkeys(_COUNTERS, 0) for name in ("team_a", "team_b")},
            )
            for totals in (self.summary["teams"][team], episode[team]):
                for name in _COUNTERS:
                    totals[name] += row.get(name, 0)
                totals[f"{row['outcome']}_actions"] += int(row["action"] is not None)
                if row["outcome"] == "played":
                    if row["source"] == "fallback":
                        totals["fallback_actions"] += 1
                    elif row["source"] == "forced":
                        totals["forced_actions"] += 1
            if call.mode != "none":
                self.sequence += 1
                self._write(
                    {
                        "type": "decision",
                        "call_id": f"{self.attempt}:{self.sequence}",
                        **row,
                    }
                )

    def flush(self) -> None:
        """Make written call evidence durable before the game writer publishes."""
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            if not self.directory_synced:
                assert self.run.directory is not None
                for path in (self.run.directory / "model_calls", self.run.directory):
                    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                self.directory_synced = True
        if self.run.publish is not None:
            self.run.publish("llm", self.attempt, self.summary)

    def completed(self, episode_ids: frozenset[int]) -> None:
        """Record game acknowledgements supplied by the sole game writer."""
        new = episode_ids - self.acknowledged
        if new and self.stream is not None:
            self._write({"type": "durable_games", "episode_ids": sorted(new)})
            self.flush()
        self.acknowledged.update(new)

    def close(self) -> None:
        """Flush and close an owned stream even if flushing raises."""
        try:
            self.flush()
        finally:
            if self.stream is not None:
                self.stream.close()
                self.stream = None


def attach(identity_json: str) -> None:
    """Register an LLM observer within an existing runner, without file creation."""
    run = current_run()
    if run is not None:
        observer = run.register("llm", lambda: _Recorder(run))
        cast(_Recorder, observer).method(identity_json)


def actor_channel(
    lane: int, actor: int, mode: RecordingMode, method_id: str
) -> CallEvidence | None:
    """Capture the runner channel before worker dispatch; direct loops return None."""
    decision, team = _CURRENT.get(), current_team()
    if decision is None:
        return None
    if team is None:
        raise RuntimeError("LLM decision lacks its shared runner team binding")
    return decision.actor(team, lane, actor, mode, method_id)


def read_calls(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream actor rows from a run directory, model_calls directory or call file.

    Each returned row includes run/pass/execution identity from its file header.
    Truncated final lines after a crash are skipped; malformed complete lines
    raise ValueError. A missing game acknowledgement is not proof the game failed:
    reconcile it with the run writer's run_details.json completed IDs. Request
    JSON exists only in full mode; its UTF-8 bytes reproduce the transmitted body.
    """
    source = Path(path)
    if source.is_dir() and (source / "run_details.json").is_file():
        source = source / "model_calls"
        if not source.exists():
            return
    paths = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    for file in paths:
        header: dict[str, Any] | None = None
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                if row.get("type") == "run":
                    header = row
                elif row.get("type") == "decision":
                    if header is None:
                        raise ValueError("Model call file has no run header")
                    yield {
                        key: header[key]
                        for key in ("run_id", "phase", "pass_id", "attempt_id")
                    } | row


def call_summary(
    result: EvaluationResult | TournamentResult | SavedResults,
) -> dict[str, Any]:
    """Return both teams' recorded LLM costs and completed-game behavior counts.

    result is an ordinary evaluation, tournament or loaded saved result. The
    selected result scope controls which passes are included. all_attempts sums
    recorded costs across successful and abandoned attempts. completed_games
    includes only the attempt that made each saved game durable. Its fallback
    counts belong beside those games' win rates. For unsaved successful runs,
    completed_games uses the current run's played decisions.

    Token totals count only reported usage; missing_usage_replies marks gaps.
    An abrupt interruption can lose unflushed costs, especially with records=none.
    These totals are not a hard spending limit. This reads existing metadata;
    it makes no model calls, reads no call files and changes nothing.
    """

    def empty() -> dict[str, dict[str, int]]:
        """Build distinct per-team numerical counters without sharing dictionaries."""
        return {name: dict.fromkeys(_COUNTERS, 0) for name in ("team_a", "team_b")}

    all_attempts, completed = empty(), empty()

    def add(target: dict[str, dict[str, int]], source: Mapping[str, Any]) -> None:
        """Sum known numerical counters, retaining zero for unrepresented teams."""
        for team, counts in target.items():
            for name in _COUNTERS:
                counts[name] += int(source.get(team, {}).get(name, 0))

    passes = cast(dict[str, Any], result.metadata.get("passes", {}))
    found = False
    for entry in passes.values():
        evidence = entry.get("host_evidence", {}).get("llm")
        if evidence is None:
            current_pass = entry.get("details", {}).get("llm")
            if result.run_dir is None and current_pass is not None:
                found = True
                add(all_attempts, current_pass["teams"])
                add(completed, current_pass["teams"])
            continue
        found = True
        attempts = evidence["attempts"]
        for attempt in attempts.values():
            add(all_attempts, attempt["teams"])
        for episode, attempt in evidence["episode_attempts"].items():
            add(completed, attempts[attempt]["episodes"][episode])
    if not found and result.run_dir is None:
        current = result.metadata.get("llm")
        if current is not None:
            current = cast(dict[str, Any], current)
            add(all_attempts, current["teams"])
            add(completed, current["teams"])
    return {
        "all_attempts": all_attempts,
        "completed_games": completed,
        "interrupted_costs_may_be_missing": True,
    }
