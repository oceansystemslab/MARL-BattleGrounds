"""Compute replay metric prefixes for the Viewer and CSV exports.

analyze_replay reads captured Core facts once and uses the same numerical
metric functions as live evaluation. Fixed-size JAX blocks limit temporary
decoding; ReplayAnalysis keeps only scalar values for each captured frame.
Seeking and exporting then select stored values without replaying game physics.
Full metrics require a matching recorded mechanics catalog. This computes the
current scalar schema from captured facts. Historical stored metric reports
keep their own recorded schemas and are not overwritten or relabelled. A
recording saved before the Red Zone rule (its context does not hold
ResolvedEnvConfigV2) shows all 44 Red Zone columns as unavailable, never as
invented zeros.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from functools import partial
from importlib.resources import files
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Reward,
)
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
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    METRIC_TOPICS,
    METRIC_VIEW_LABELS,
    PRIORITY_METRIC_COLUMNS,
    STATUS_LABELS,
    MetricColumn,
    metric_guidance,
    metric_locations,
    metric_order_key,
    metric_primary_location,
    metric_search_facts,
    metric_search_terms,
    metric_topic_text,
)
from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeCompletionV1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContext,
    ResolvedEnvConfigV2,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle

type MetricScope = Literal["cursor", "final"]
# A common shape reuses one compiled scan across replay lengths. Host decoding is
# bounded by this block, rather than another full copy of the recorded trajectory.
_BLOCK_SIZE = 64
# Full-catalog positions of the 44 Red Zone columns. A recording without a
# recorded Red Zone rule marks exactly these unavailable.
_RED_ZONE_COLUMN_INDICES = np.asarray(
    [
        index
        for index, column in enumerate(METRIC_COLUMNS)
        if column.family == "red_zone"
    ]
)
REPLAY_IDENTITY_COLUMNS = (
    "episode_id",
    "scope",
    "frame_index",
    "simulator_step_count",
    "metric_schema_id",
    "metric_schema_version",
    "source_replay_digest",
    "analysis_source_digest",
    "completion_state",
    *(
        f"agent_{slot}_{field}"
        for slot in range(10)
        for field in ("active", "class_id", "class", "policy_id")
    ),
)


class _Carry(NamedTuple):
    """Carry the last captured state and compact metric totals between analysis
    blocks.
    """

    state: EnvState
    priority: PriorityTotals
    diagnostics: FullTotals


class _Transition(NamedTuple):
    """One decoded successor, acting mask, Core facts and reward, with an admission
    flag.
    """

    successor: EnvState
    action_mask: ActionMask
    info: Info
    reward: Reward
    valid: Array


def _source_digest() -> str:
    """Hash installed evaluation/Core source bytes so analyzed values identify their
    producer.
    """
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
    """Read current priority values, adding full values only for the static full
    mode.
    """
    priority = priority_values(
        config, carry.state, initial_step, carry.priority, outcome
    )
    return full_values(carry.diagnostics, config, priority) if full else priority


@partial(jax.jit, static_argnames=("full",))
def _initialize(
    config: EnvConfig, state: EnvState, *, full: bool
) -> tuple[_Carry, MetricValues]:
    """Create one prefix's metric totals and values at the captured initial frame."""
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
    """Reduce one fixed-size captured block without advancing simulator physics.

    full is static; config/state/facts are numerical inputs. Valid rows update the
    shared metric authorities and return prefix values. Padding leaves carry
    unchanged and returns unavailable zeros that the host removes.
    """

    def step(previous: _Carry, row: _Transition) -> tuple[_Carry, MetricValues]:
        """Choose a real recorded transition or inert padding while keeping fixed
        shapes.
        """

        def observed(_: None) -> tuple[_Carry, MetricValues]:
            """Update totals from captured facts and read the successor's metric
            prefix.
            """
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
            """Preserve carry and emit unavailable zero values for a padded block
            row.
            """
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
    """Read stored metric prefixes without recomputing a recorded game.

    Attributes
    ----------
    source_replay_digest : str
        Digest of the analyzed replay artifact.
    analysis_source_digest : str
        Digest of installed evaluation/Core source bytes.
    original_metric_status : str
        Status of any original sidecar: not_recorded,
        missing, available or empty. Current values are computed separately.
    context : EvaluationEpisodeContext
        Immutable recorded episode/config/information context.
    completion : EvaluationEpisodeCompletionV1
        Recorded completion description for the final captured frame.
    columns : tuple[MetricColumn, ...]
        Ordered metric catalog entries represented by the stored arrays.
    frame_count : object
        Number of captured frames, including the initial frame.

    Notes
    -----
    Construct through analyze_replay. Internal float32 values and boolean
    validity arrays have shape (frames, columns) and are read-only. This frozen
    descriptor also caches nested Python display metadata; treat returned
    catalog/topic descriptions as read-only. No method owns a file handle.
    """

    source_replay_digest: str
    analysis_source_digest: str
    original_metric_status: str
    context: EvaluationEpisodeContext
    completion: EvaluationEpisodeCompletionV1
    columns: tuple[MetricColumn, ...]
    _step_counts: tuple[int, ...]
    _values: NDArray[np.float32]
    _valid: NDArray[np.bool_]
    _row_templates: tuple[dict[str, object], ...] = field(init=False, repr=False)
    _not_applicable_reasons: tuple[str | None, ...] = field(init=False, repr=False)
    _topics: tuple[dict[str, object], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Cache roster-aware descriptions once, without choosing cursor time or POV."""
        rows: list[dict[str, object]] = []
        reasons: list[str | None] = []
        recorded_classes = tuple(agent.class_id for agent in self.context.roster)
        class_ids = tuple(
            agent.class_id if agent.configured_active else 0
            for agent in self.context.roster
        )
        for order, column in enumerate(self.columns):
            direction, guidance = metric_guidance(column, recorded_classes)
            primary_topic, primary_view = metric_primary_location(column)
            locations = (
                metric_locations(column, class_ids)
                if len(self.columns) > len(PRIORITY_METRIC_COLUMNS)
                else ((primary_topic, primary_view),)
            )
            reason = self._not_applicable_reason(column)
            reasons.append(reason)
            topic_text = {
                topic: text
                for topic, _ in locations
                if (text := metric_topic_text(column, topic, class_ids))
            }
            rows.append(
                {
                    "name": column.name,
                    "stem": column.stem,
                    "label": column.label,
                    "family": column.family,
                    "primary_topic": primary_topic,
                    "primary_view": primary_view,
                    "locations": [
                        {"topic": topic, "view": view} for topic, view in locations
                    ],
                    "topic_order": {
                        topic: list(metric_order_key(column, topic))
                        for topic, _ in locations
                    },
                    "unit": column.unit,
                    "status": None
                    if column.status_channel is None
                    else STATUS_LABELS[column.status_channel],
                    "scope": column.scope,
                    "subjects": column.subjects,
                    "subject": self._subject(column),
                    "subject_role": column.subject_role,
                    "recipient_role": column.recipient_role,
                    "description": column.description,
                    "numerator": column.numerator,
                    "denominator": column.denominator,
                    "direction": direction,
                    "guidance": guidance,
                    "missing_when": column.missing_when,
                    "applicable": reason is None,
                    "order": order,
                    **({"topic_text": topic_text} if topic_text else {}),
                }
            )
        object.__setattr__(self, "_row_templates", tuple(rows))
        object.__setattr__(self, "_not_applicable_reasons", tuple(reasons))
        object.__setattr__(
            self,
            "_topics",
            tuple(
                {
                    "name": topic.name,
                    "label": topic.label,
                    "section": topic.section,
                    "description": topic.description,
                    "views": (
                        [
                            {"name": name, "label": METRIC_VIEW_LABELS[name]}
                            for name in ("totals", "recipients")
                        ]
                        if topic.paired
                        else [{"name": "single", "label": METRIC_VIEW_LABELS["single"]}]
                    ),
                }
                for topic in METRIC_TOPICS
                if (
                    len(self.columns) > len(PRIORITY_METRIC_COLUMNS)
                    or topic.name == "priority"
                )
            ),
        )

    def _not_applicable_reason(self, column: MetricColumn) -> str | None:
        """Explain a structural roster/targeting limit for one display column.

        Use configured active classes and recorded mechanics, never a current value,
        cooldown or zero denominator. Return None when the column can apply. This
        display filter does not change the numerical CSV availability mask.
        """
        roster = self.context.roster
        if column.scope == "episode":
            return None
        if column.scope in ("team", "team_recipient"):
            sources = tuple(
                agent.global_slot
                for agent in roster
                if agent.configured_team_id == column.subjects[0]
                and agent.configured_active
                and (
                    column.required_class_id is None
                    or agent.class_id == column.required_class_id
                )
            )
            if not sources:
                team = "A" if column.subjects[0] == 1 else "B"
                if column.required_class_id is not None:
                    name = self.context.static_mechanics_catalog.class_name_by_id[
                        column.required_class_id
                    ]
                    return f"This measurement needs an active {name} on Team {team}."
                return f"Team {team} has no active agents in this replay."
        else:
            for slot in column.subjects:
                if not roster[slot].configured_active:
                    return f"Agent ID {slot} is inactive in this replay."
            source = column.subjects[0]
            if (
                column.required_class_id is not None
                and roster[source].class_id != column.required_class_id
            ):
                name = self.context.static_mechanics_catalog.class_name_by_id[
                    column.required_class_id
                ]
                return f"Agent ID {source} must be a {name} for this measurement."
            sources = (source,)
        reason = self._effect_not_applicable_reason(column, sources)
        if reason is not None:
            return reason
        if column.scope not in ("source_recipient", "team_recipient"):
            return None
        recipient = column.subjects[1]
        if not roster[recipient].configured_active:
            return f"Agent ID {recipient} is inactive in this replay."
        mechanics = self.context.static_mechanics_catalog.class_mechanics
        if not (column.requires_basic_target or column.requires_ultimate_target):
            return None
        for source in sources:
            profile = mechanics[roster[source].class_id]
            mode = (
                profile.basic_target_mode
                if column.requires_basic_target
                else profile.ultimate_target_mode
            )
            same_team = (
                roster[source].configured_team_id
                == roster[recipient].configured_team_id
            )
            if (
                (mode == "target_none" and source == recipient)
                or (mode == "ally" and same_team)
                or (mode == "enemy" and not same_team)
            ):
                return None
        ability = "Basic ability" if column.requires_basic_target else "Ultimate"
        if column.scope == "team_recipient":
            return (
                f"The agents counted by this measurement cannot use this {ability} "
                f"on Agent ID {recipient}."
            )
        return f"Agent ID {sources[0]}'s {ability} cannot target Agent ID {recipient}."

    def _effect_not_applicable_reason(
        self, column: MetricColumn, sources: tuple[int, ...]
    ) -> str | None:
        """Explain impossible effect producers using recorded class abilities.

        sources contains configured global slots for the metric subject. Recipients
        are not mistaken for their helpers. Preserve rows for capable agents before
        they act, and preserve authored Trap intervals without inventing a caster.
        Red Zone kill help uses the same rule as other kill help: a team needs an
        agent who can deal damage, or a Priest with a damaging teammate. Red Zone
        deaths belong to their victims and always apply.
        This display-only explanation never changes numerical values or CSV masks.
        """
        health_effect = (
            "damage"
            if column.family
            in (
                "damage_done",
                "damage_received",
                "recipient_damage",
                "controlled_damage",
            )
            else "healing"
            if column.family
            in (
                "healing_done",
                "healing_received",
                "recipient_healing",
                "excess_healing",
                "controlled_healing",
            )
            else None
        )
        if health_effect is None and column.family not in (
            "priest_rescue",
            "burst",
            "aura_coverage",
            "aura_benefits",
            "poison",
            "trap_breaks",
            "kill_contributions",
            "controlled_kills",
            "red_zone",
            "coordination",
            "formation",
        ):
            return None
        roster = self.context.roster
        mechanics = self.context.static_mechanics_catalog.class_mechanics
        team = roster[sources[0]].configured_team_id
        team_name = "A" if team == 1 else "B"
        allies = tuple(
            agent.global_slot
            for agent in roster
            if agent.configured_active and agent.configured_team_id == team
        )
        opponents = tuple(
            agent.global_slot
            for agent in roster
            if agent.configured_active and agent.configured_team_id != team
        )

        def has_class(team_id: int, class_id: int) -> bool:
            """Check active recorded roster membership for a configured team and
            class.
            """
            return any(
                agent.configured_active
                and agent.configured_team_id == team_id
                and agent.class_id == class_id
                for agent in roster
            )

        def can_produce(slot: int, effect: str = "damage", ability: str = "") -> bool:
            """Check recorded raw Basic/Ultimate effect capability for one source
            slot.
            """
            profile = mechanics[roster[slot].class_id]
            return (
                ability != "ultimate" and getattr(profile, f"basic_raw_{effect}") > 0
            ) or (ability != "basic" and getattr(profile, f"ultimate_raw_{effect}") > 0)

        def team_can_damage(team_id: int) -> bool:
            """Check whether any active recorded teammate can produce damage."""
            return any(
                agent.configured_active
                and agent.configured_team_id == team_id
                and can_produce(agent.global_slot)
                for agent in roster
            )

        if health_effect is not None:
            ability = (
                "basic"
                if column.stem.startswith("basic_")
                else "ultimate"
                if column.stem.startswith("ultimate_")
                else ""
            )
            recipient_owned = column.subject_role == "recipient"
            producers = (
                (opponents if health_effect == "damage" else allies)
                if recipient_owned
                else sources
            )
            if column.family == "healing_received":
                regeneration = any(
                    mechanics[
                        roster[slot].class_id
                    ].out_of_combat_health_regeneration_fraction_per_step
                    > 0
                    for slot in sources
                )
                if column.stem in (
                    "regenerated_healing",
                    "regeneration_healing_received_fraction",
                ):
                    return (
                        None
                        if regeneration
                        else "These agents cannot regenerate health."
                    )
                if regeneration and column.stem in (
                    "healing_received",
                    "healing_received_fraction",
                    "effective_healing_received",
                ):
                    return None
            producers = tuple(
                source
                for source in producers
                if can_produce(source, health_effect, ability)
            )
            if column.family == "controlled_healing" and column.status_channel in (
                3,
                4,
                5,
            ):
                # A Priest cannot act while stunned. Another Priest can still
                # heal it; other classes can still be the stunned patient.
                target = (
                    column.subjects[1]
                    if column.scope in ("source_recipient", "team_recipient")
                    else column.subjects[0]
                    if recipient_owned and column.scope == "agent"
                    else None
                )
                producers = tuple(
                    source
                    for source in producers
                    if (source != target if target is not None else len(allies) > 1)
                )
                if not producers:
                    return (
                        "An already-stunned agent needs a Priest teammate to heal it."
                    )
            if not producers:
                verb = "deal" if health_effect == "damage" else "provide"
                if recipient_owned:
                    from_team = "B" if team == 1 else "A"
                    if health_effect == "healing":
                        from_team = team_name
                    return (
                        f"Team {from_team} has no active agent that can {verb} "
                        f"{ability.title() + ' ' if ability else ''}{health_effect}."
                    )
                who = (
                    f"Team {team_name}"
                    if column.scope in ("team", "team_recipient")
                    else f"Agent ID {sources[0]}"
                )
                ability_name = ability.title() if ability else "Basic or Ultimate"
                return (
                    f"{who} cannot {verb} this {health_effect} "
                    f"with {ability_name} abilities."
                )
            return None
        if column.family == "priest_rescue":
            if not has_class(team, PRIEST_CLASS_ID):
                return f"Team {team_name} has no active Priest to save this ally."
            if column.subject_role != "recipient" and not any(
                roster[source].class_id == PRIEST_CLASS_ID for source in sources
            ):
                return f"Agent ID {sources[0]} must be a Priest to help with healing."
            if not team_can_damage(3 - team):
                return (
                    "No active opponent can deal the damage counted by a healing save."
                )
        elif column.family == "burst" and column.subject_role == "recipient":
            if not has_class(3 - team, MAGE_CLASS_ID):
                return "The opposing team has no active Mage to deal Burst damage."
        elif column.family == "aura_coverage" and column.subject_role == "recipient":
            class_id = (
                MAGE_CLASS_ID if column.stem.startswith("mage_") else WARRIOR_CLASS_ID
            )
            if not has_class(team, class_id):
                name = self.context.static_mechanics_catalog.class_name_by_id[class_id]
                return f"Team {team_name} has no active {name} to give this aura."
        elif column.family == "poison" and not has_class(team, PRIEST_CLASS_ID):
            return (
                f"Team {team_name} has no active Priest "
                "whose healing Poison could reduce."
            )
        elif column.stem == "damage_prevented_by_warrior_aura":
            if not team_can_damage(3 - team):
                return (
                    "No active opponent can deal damage for the Warrior aura to reduce."
                )
        elif column.family == "formation" and len(allies) < 2:
            return (
                f"Team {team_name} needs two active agents to measure their distance."
            )
        elif column.family == "coordination":
            damagers = sum(can_produce(slot) for slot in allies)
            if column.stem.startswith("focus_fire_") and damagers < 2:
                return "Focus fire needs at least two teammates who can deal damage."
            if column.stem.startswith("single_contributor_") and not damagers:
                return "A single-contributor kill needs an agent who can deal damage."
            if column.stem.startswith("multi_contributor_") and not (
                damagers >= 2 or (damagers and has_class(team, PRIEST_CLASS_ID))
            ):
                return (
                    "A shared kill needs two attackers, or an attacker "
                    "and a Priest who helps through healing."
                )

        if column.family == "trap_breaks":
            # Initial Traps remain observable without a caster or breaker.
            if column.stem == "trap_intervals":
                return None
            breakers = opponents if column.subject_role == "recipient" else sources
            if not any(can_produce(source) for source in breakers):
                who = (
                    "The opposing team"
                    if column.subject_role == "recipient"
                    else f"Team {team_name}"
                    if column.scope == "team"
                    else f"Agent ID {sources[0]}"
                )
                return f"{who} cannot deal damage to break a Trap."

        if column.subject_role == "recipient":
            return None
        if column.family in ("kill_contributions", "controlled_kills", "red_zone"):
            solo = column.stem.startswith("solo_")
            ability = (
                "ultimate"
                if "ultimate_" in column.stem
                else "basic"
                if "basic_" in column.stem
                else ""
            )
            for source in sources:
                if can_produce(source, ability=ability):
                    return None
                if (
                    not solo
                    and roster[source].class_id == PRIEST_CLASS_ID
                    and team_can_damage(team)
                ):
                    # The healed ally attacks the enemy. The Priest does not
                    # need an enemy-targeting ability to earn support credit.
                    return None
            if solo:
                return "A Priest's healing support cannot earn a solo kill."
            if ability == "ultimate" and any(
                roster[source].class_id == MAGE_CLASS_ID for source in sources
            ):
                return (
                    "Activating Burst does not deal damage "
                    "or earn Ultimate kill credit."
                )
            return f"Team {team_name} has no damaging teammate for Priest kill support."
        return None

    def catalog(self) -> dict[str, object]:
        """Return metric descriptions and display navigation without choosing a frame.

        Returns
        -------
        dict[str, object]
            Dict with schema/source identities, topics, classes, global agent rows and
            every selected column's meaning, applicability and search terms.

        Notes
        -----
        Host-only; reads cached metadata and performs no numerical analysis.
        Treat nested descriptions as read-only because some are shared with this
        analysis object. This catalog contains no current-frame metric values.
        """
        class_names = self.context.static_mechanics_catalog.class_name_by_id
        return {
            "metric_schema_id": METRIC_SCHEMA_ID,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "source_replay_digest": self.source_replay_digest,
            "analysis_source_digest": self.analysis_source_digest,
            "topics": self._topics,
            "class_names": class_names,
            "agents": [
                {
                    "slot": agent.global_slot,
                    "class_id": agent.class_id,
                    "class_name": class_names[agent.class_id],
                    "team_id": 1 if agent.global_slot < 5 else 2,
                    "active": agent.configured_active,
                }
                for agent in self.context.roster
            ],
            "measurements": [
                {
                    **row,
                    "not_applicable_reason": reason,
                    "search_terms": metric_search_terms(column),
                    "search_facts": metric_search_facts(column),
                }
                for column, row, reason in zip(
                    self.columns,
                    self._row_templates,
                    self._not_applicable_reasons,
                    strict=True,
                )
            ],
        }

    @property
    def frame_count(self) -> int:
        """Return initial-frame count plus the number of captured real transitions."""
        return len(self._values)

    def _frame(self, frame_index: int, scope: MetricScope) -> int:
        """Validate cursor/scope and choose the requested or final cached frame."""
        if scope not in ("cursor", "final"):
            raise ValueError("metric scope must be cursor or final")
        if type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise IndexError("metric cursor is outside the captured replay")
        return self.frame_count - 1 if scope == "final" else frame_index

    def _metadata(self, selected: int, scope: MetricScope) -> dict[str, object]:
        """Build selected-prefix identity fields from recorded context and frame
        time.
        """
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
        """Describe a metric's recorded global agent/team subjects in display order."""
        if column.scope == "episode":
            return "Episode"
        if column.scope == "team":
            return "Team A" if column.subjects[0] == 1 else "Team B"

        def agent(slot: int) -> str:
            """Name a global slot by recorded class and fixed Team A/Team B
            ownership.
            """
            roster = self.context.roster[slot]
            class_name = self.context.static_mechanics_catalog.class_name_by_id[
                roster.class_id
            ]
            # Inactive slots may carry neutral team IDs; slot ownership is fixed.
            team = "A" if slot < 5 else "B"
            return f"Agent ID {slot} · {class_name} · Team {team}"

        if column.scope == "agent":
            return agent(column.subjects[0])
        if column.scope == "team_recipient":
            team = "A" if column.subjects[0] == 1 else "B"
            return f"Team {team} → {agent(column.subjects[1])}"
        separator = " → " if column.scope == "source_recipient" else " ↔ "
        return separator.join(agent(slot) for slot in column.subjects)

    def summary(
        self, frame_index: int, *, scope: MetricScope = "cursor"
    ) -> dict[str, object]:
        """Read described metric values at a captured frame or the final boundary.

        Parameters
        ----------
        frame_index : int
            Integer captured-frame index in 0..frame_count-1, including
            the initial frame at zero. It must be valid even for final scope.
        scope : MetricScope
            "cursor" by default, or "final" to select the last captured frame.

        Returns
        -------
        dict[str, object]
            Host dict with prefix identity, topics and one statistics row per column.
            Unavailable values are None with valid=False. completion is included only
            at the final captured frame; earlier prefixes are labeled partial.

        Raises
        ------
        ValueError
            scope is neither cursor nor final.
        IndexError
            frame_index is not an exact in-range Python int.

        Notes
        -----
        No simulator or metric kernel runs here. Values come from the cached
        prefix arrays. Treat nested shared description fields as read-only.
        """
        selected = self._frame(frame_index, scope)
        rows: list[dict[str, object]] = []
        for index, template in enumerate(self._row_templates):
            valid = bool(self._valid[selected, index])
            rows.append(
                {
                    **template,
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
            "topics": self._topics,
            "statistics": rows,
        }

    def csv(self, frame_index: int, *, scope: MetricScope = "cursor") -> str:
        """Return one wide CSV row using the same cached boundary as summary.

        Parameters
        ----------
        frame_index : int
            Exact Python int in 0..frame_count-1; required even for final.
        scope : MetricScope
            "cursor" by default, or "final" for the last captured frame.

        Returns
        -------
        str
            CSV text containing a header and one data row. Identity columns precede
            the ordered selected metrics. Unavailable values become empty cells;
            valid zero remains numeric zero.

        Raises
        ------
        ValueError
            scope is unknown.
        IndexError
            frame_index is not an exact in-range Python int.

        Notes
        -----
        Host-only and no file I/O. The caller owns saving the returned string.
        Export uses the recorded prefix; it does not rerun metrics or game physics.
        """
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
        writer = csv.DictWriter(
            output,
            fieldnames=(
                *REPLAY_IDENTITY_COLUMNS,
                *(column.name for column in self.columns),
            ),
        )
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue()


def analyze_replay(bundle: LoadedReplayBundle, *, full: bool = False) -> ReplayAnalysis:
    """Compute scalar metric values for every captured replay frame.

    Parameters
    ----------
    bundle : LoadedReplayBundle
        LoadedReplayBundle with validated captured state, transition facts
        and context. Any historical metric sidecar remains separate evidence.
    full : bool
        False by default for priority metrics only. True adds the current
        full metric catalog and requires matching recorded mechanics.

    Returns
    -------
    ReplayAnalysis
        ReplayAnalysis with read-only float32 values and boolean validity for
        initial frame plus every captured transition. Column order is current
        PRIORITY_METRIC_COLUMNS or METRIC_COLUMNS, independent of old sidecars.

    Raises
    ------
    ValueError
        The recorded task mode is outside neutral/TDM (0/1), or full
        analysis sees a different mechanics catalog.
    RuntimeError
        Evaluation/Core source bytes change during analysis.
    TypeError
        Captured data cannot be reconstructed with its declared types.

    Notes
    -----
    Host orchestration decodes at most 64 transitions per block and calls JAX
    metric reducers. It may compile and transfer arrays, but never reruns the
    simulator. Full metrics reuse Core's existing counterfactual helpers.
    The final short block uses inert padding that creates no extra prefix.
    No files are written. Later summary/csv calls reuse the stored scalars;
    source-byte digests intentionally change when source documentation changes.
    With full=True, a context without ResolvedEnvConfigV2 (recorded before the
    Red Zone rule) has all 44 Red Zone columns marked unavailable in every
    frame; every older column keeps its computed value and availability.
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
    if full and not isinstance(context.resolved_env_config, ResolvedEnvConfigV2):
        # No Red Zone rule was recorded; do not report reconstructed depth-0
        # zeros as evidence.
        valid[:, _RED_ZONE_COLUMN_INDICES] = False
    scalars.setflags(write=False)
    valid.setflags(write=False)
    if _source_digest() != digest:
        raise RuntimeError("evaluation source changed during replay analysis")
    original = bundle.metric_report_artifact
    columns = METRIC_COLUMNS if full else PRIORITY_METRIC_COLUMNS
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
        columns=columns,
        _step_counts=tuple(frame.simulator_step_count for frame in replay.frames),
        _values=scalars,
        _valid=valid,
    )
