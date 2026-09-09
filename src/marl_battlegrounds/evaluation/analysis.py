"""Read-only replay prefixes using the training/evaluation scalar metric authority.

Decode captured Core arrays once, scan fixed-size blocks, and retain only scalar
prefixes. Seeking and CSV export never rerun the simulator or metric collection.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from functools import partial
from importlib.resources import files
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from marl_battlegrounds.core.types import ActionMask, EnvConfig, EnvState, Info, Reward
from marl_battlegrounds.evaluation.capture import (
    reconstruct_env_state_v1,
    reconstruct_transition_facts_v1,
)
from marl_battlegrounds.evaluation.catalog import (
    build_static_mechanics_catalog_v1,
    reconstruct_env_config_v1,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
    PriorityTotals,
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.full_metrics import (
    FullTotals,
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    METRIC_FAMILIES,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_COLUMNS,
    STATUS_LABELS,
    MetricColumn,
)
from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeCompletionV1
from marl_battlegrounds.evaluation.models import EvaluationEpisodeContext
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle

type MetricScope = Literal["cursor", "final"]
# A common shape reuses one compiled scan across replay lengths. Host decoding is
# bounded by this block, rather than another full copy of the recorded trajectory.
_BLOCK_SIZE = 64


class _Carry(NamedTuple):
    state: EnvState
    priority: PriorityTotals
    diagnostics: FullTotals


class _Transition(NamedTuple):
    successor: EnvState
    action_mask: ActionMask
    info: Info
    reward: Reward
    valid: Array


def _source_digest() -> str:
    digest = hashlib.sha256()
    for package in ("marl_battlegrounds.evaluation", "marl_battlegrounds.core"):
        for path in sorted(files(package).iterdir(), key=lambda path: path.name):
            if path.name.endswith(".py"):
                digest.update(f"{package}/{path.name}\0".encode())
                digest.update(path.read_bytes())
                digest.update(b"\0")
    return digest.hexdigest()


def _values(
    config: EnvConfig, initial_step: Array, carry: _Carry, outcome: Array, *, full: bool
) -> MetricValues:
    priority = priority_values(
        config, carry.state, initial_step, carry.priority, outcome
    )
    return full_values(carry.diagnostics, config, priority) if full else priority


@partial(jax.jit, static_argnames=("full",))
def _initialize(
    config: EnvConfig, state: EnvState, *, full: bool
) -> tuple[_Carry, MetricValues]:
    carry = _Carry(
        state, initialize_priority(), initialize_full(config, state) if full else {}
    )
    return carry, _values(
        config, state.step_count, carry, jnp.asarray(0, jnp.int32), full=full
    )


@partial(jax.jit, static_argnames=("full",))
def _scan_block(
    config: EnvConfig,
    initial_step: Array,
    carry: _Carry,
    rows: _Transition,
    *,
    full: bool,
) -> tuple[_Carry, MetricValues]:
    def step(previous: _Carry, row: _Transition) -> tuple[_Carry, MetricValues]:
        def observed(_: None) -> tuple[_Carry, MetricValues]:
            current = _Carry(
                row.successor,
                update_priority(previous.priority, row.reward, row.info),
                update_full(
                    previous.diagnostics,
                    config,
                    previous.state,
                    row.action_mask,
                    row.info,
                )
                if full
                else previous.diagnostics,
            )
            return current, _values(
                config,
                initial_step,
                current,
                row.info.transition_facts.team_deathmatch_facts.outcome,
                full=full,
            )

        def padding(_: None) -> tuple[_Carry, MetricValues]:
            size = len(METRIC_COLUMNS) if full else len(PRIORITY_METRIC_COLUMNS)
            return previous, MetricValues(
                jnp.zeros(size, jnp.float32), jnp.zeros(size, bool)
            )

        return cast(
            tuple[_Carry, MetricValues],
            jax.lax.cond(row.valid, observed, padding, None),
        )

    return jax.lax.scan(step, carry, rows)


@dataclass(frozen=True, slots=True)
class ReplayAnalysis:
    """Immutable scalar prefixes; GUI and CSV select the identical stored boundary."""

    source_replay_digest: str
    analysis_source_digest: str
    original_metric_status: str
    context: EvaluationEpisodeContext
    completion: EvaluationEpisodeCompletionV1
    columns: tuple[MetricColumn, ...]
    _step_counts: tuple[int, ...]
    _values: NDArray[np.float32]
    _valid: NDArray[np.bool_]

    @property
    def frame_count(self) -> int:
        return len(self._values)

    def _frame(self, frame_index: int, scope: MetricScope) -> int:
        if scope not in ("cursor", "final"):
            raise ValueError("metric scope must be cursor or final")
        if type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise IndexError("metric cursor is outside the captured replay")
        return self.frame_count - 1 if scope == "final" else frame_index

    def _metadata(self, selected: int, scope: MetricScope) -> dict[str, object]:
        context = self.context
        metadata: dict[str, object] = {
            "episode_id": context.identity.episode_id,
            "scope": scope,
            "frame_index": selected,
            "simulator_step_count": self._step_counts[selected],
            "metric_schema_id": METRIC_SCHEMA_ID,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "source_replay_digest": self.source_replay_digest,
            "analysis_source_digest": self.analysis_source_digest,
            "completion_state": self.completion.completion_state
            if selected == self.frame_count - 1
            else "partial",
        }
        for roster, assignment in zip(
            context.roster, context.policy_assignments, strict=True
        ):
            prefix = f"agent_{roster.global_slot}"
            metadata[f"{prefix}_active"] = roster.configured_active
            metadata[f"{prefix}_class_id"] = roster.class_id
            metadata[f"{prefix}_class"] = (
                self.context.static_mechanics_catalog.class_name_by_id[roster.class_id]
            )
            metadata[f"{prefix}_policy_id"] = (
                assignment.policy_id
                if assignment.assignment_status == "assigned"
                else None
            )
        return metadata

    def _subject(self, column: MetricColumn) -> str:
        if column.scope == "episode":
            return "Episode"
        if column.scope == "team":
            return "Team A" if column.subjects[0] == 1 else "Team B"

        def agent(slot: int) -> str:
            roster = self.context.roster[slot]
            class_name = self.context.static_mechanics_catalog.class_name_by_id[
                roster.class_id
            ]
            return f"Agent {slot} · {class_name}"

        if column.scope == "agent":
            return agent(column.subjects[0])
        separator = " → " if column.scope == "source_recipient" else " ↔ "
        return separator.join(agent(slot) for slot in column.subjects)

    def summary(
        self, frame_index: int, *, scope: MetricScope = "cursor"
    ) -> dict[str, object]:
        """Return catalog descriptions and scalar values, never GUI-side formulas."""
        selected = self._frame(frame_index, scope)
        rows: list[dict[str, object]] = []
        for index, column in enumerate(self.columns):
            valid = bool(self._valid[selected, index])
            rows.append(
                {
                    "name": column.name,
                    "label": column.label,
                    "family": column.family,
                    "unit": column.unit,
                    "status": None
                    if column.status_channel is None
                    else STATUS_LABELS[column.status_channel],
                    "scope": column.scope,
                    "subjects": column.subjects,
                    "subject": self._subject(column),
                    "description": column.description,
                    "direction": column.direction,
                    "missing_when": column.missing_when,
                    "value": float(self._values[selected, index]) if valid else None,
                    "valid": valid,
                }
            )
        return {
            **self._metadata(selected, scope),
            "captured_transition_count": self.frame_count - 1,
            "original_metric_status": self.original_metric_status,
            "completion": self.completion.model_dump(mode="json")
            if selected == self.frame_count - 1
            else None,
            "families": [
                {
                    "name": name,
                    "label": METRIC_FAMILIES[name][0],
                    "description": METRIC_FAMILIES[name][1],
                }
                for name in (
                    "overview",
                    *dict.fromkeys(column.family for column in self.columns),
                )
            ],
            "statistics": rows,
        }

    def csv(self, frame_index: int, *, scope: MetricScope = "cursor") -> str:
        """One wide episode row; unavailable scalar values are empty CSV cells."""
        selected = self._frame(frame_index, scope)
        row = self._metadata(selected, scope)
        row.update(
            {
                column.name: float(self._values[selected, index])
                if self._valid[selected, index]
                else None
                for index, column in enumerate(self.columns)
            }
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue()


def analyze_replay(bundle: LoadedReplayBundle, *, full: bool = False) -> ReplayAnalysis:
    """Read recorded arrays once; priority metrics default, full diagnostics opt in.

    This is an analysis of captured facts under the installed scalar metric
    schema, independent of any historical V1 sidecar. Full diagnostics reuse Core
    counterfactual helpers, so their recorded mechanics catalog must match.
    """
    replay = bundle.replay
    context = replay.header.context
    if context.resolved_env_config.task_mode not in (0, 1):
        raise ValueError("scalar replay analysis supports TDM only")
    if full and context.static_mechanics_catalog != build_static_mechanics_catalog_v1():
        raise ValueError(
            "full analysis requires the recorded mechanics catalog version"
        )
    digest = _source_digest()
    config = reconstruct_env_config_v1(context)
    initial = reconstruct_env_state_v1(replay.frames[0], host=True)
    carry, first = cast(
        tuple[_Carry, MetricValues], _initialize(config, initial, full=full)
    )
    first = jax.device_get(first)
    value_blocks = [np.asarray(first.values)[None]]
    valid_blocks = [np.asarray(first.valid)[None]]
    initial_step = jnp.asarray(initial.step_count)
    for start in range(0, len(replay.transitions), _BLOCK_SIZE):
        transitions = replay.transitions[start : start + _BLOCK_SIZE]
        rows: list[_Transition] = []
        for offset, transition in enumerate(transitions):
            frame_index = start + offset
            frame = replay.frames[frame_index]
            rows.append(
                _Transition(
                    reconstruct_env_state_v1(replay.frames[frame_index + 1], host=True),
                    ActionMask(
                        *(
                            cast(
                                Array,
                                np.asarray(
                                    getattr(frame.action_mask, name), dtype=bool
                                ),
                            )
                            for name in ActionMask._fields
                        )
                    ),
                    Info(reconstruct_transition_facts_v1(transition.facts, host=True)),
                    Reward(
                        cast(
                            Array,
                            np.asarray(
                                transition.canonical_reward_by_agent, dtype=np.float32
                            ),
                        )
                    ),
                    cast(Array, np.asarray(True)),
                )
            )
        # Padding reuses a row only as inert storage; the validity branch cannot
        # update a counter, advance time, or create an extra prefix.
        padding = rows[-1]._replace(valid=cast(Array, np.asarray(False)))
        rows.extend([padding] * (_BLOCK_SIZE - len(rows)))
        batch = jax.tree.map(lambda *leaves: np.stack(leaves), *rows)
        carry, values = cast(
            tuple[_Carry, MetricValues],
            _scan_block(config, initial_step, carry, batch, full=full),
        )
        values = jax.device_get(values)
        value_blocks.append(np.asarray(values.values)[: len(transitions)])
        valid_blocks.append(np.asarray(values.valid)[: len(transitions)])
    scalars = np.concatenate(value_blocks)
    valid = np.concatenate(valid_blocks)
    scalars.setflags(write=False)
    valid.setflags(write=False)
    if _source_digest() != digest:
        raise RuntimeError("evaluation source changed during replay analysis")
    original = bundle.metric_report_artifact
    return ReplayAnalysis(
        source_replay_digest=replay.canonical_digest_sha256,
        analysis_source_digest=digest,
        original_metric_status="not_recorded"
        if bundle.status == "not_recorded"
        else "missing"
        if original is None
        else "available"
        if original.report.statistics
        else "empty",
        context=context,
        completion=replay.completion,
        columns=METRIC_COLUMNS if full else PRIORITY_METRIC_COLUMNS,
        _step_counts=tuple(frame.simulator_step_count for frame in replay.frames),
        _values=scalars,
        _valid=valid,
    )
