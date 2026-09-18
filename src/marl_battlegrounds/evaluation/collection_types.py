"""Describe private bounded records shared by collection and host recording.

These numerical NamedTuples transport existing measurements and evidence. They
do not define another report schema. Device buffers have fixed capacity; counts
select occupied prefixes. A drained batch has already been sliced to those
prefixes. Config indices refer only to that batch's evidence table. The writer
checks identities and publishes records; none of these types is actor input.
"""

from typing import NamedTuple

from jax import Array

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets


class ConfigEvidence(NamedTuple):
    """Keep one actual producing configuration per episode within a drain.

    episode_id and decision_step are int32 (R,); decision_step is the first
    observed action index, possibly -1 for terminal padding. config has leading
    R on every leaf. Occupied rows include episodes without emitted records.
    This table is temporary evidence, not a source bank or durable identity.
    """

    episode_id: Array
    decision_step: Array
    config: EnvConfig


class CollectedStarts(NamedTuple):
    """Pair compact source declarations with their original transition evidence.

    records contains the existing EpisodeStartRecords with leading R. The other
    fields are int32 (R,): the producing info ID, its action index and the local
    evidence-table index. Keeping original IDs prevents a false claim from
    replacing the evidence against which it must be checked.
    """

    records: EpisodeStartRecords
    info_episode_id: Array
    decision_step: Array
    config_index: Array


class CollectedCompletions(NamedTuple):
    """Retain exact terminal facts and references to available measurements.

    Every field is int32 (R,), except team_scores which is (R,2), Team A first.
    config_index selects actual evidence. priority_index and full_index select
    their separate metric tables; -1 means absent. When full_index is present,
    its catalog prefix also supplies priority values without a duplicate row.
    Required integer facts remain exact even above float32's integer precision.
    """

    episode_id: Array
    outcome: Array
    decision_step: Array
    episode_length: Array
    team_scores: Array
    config_index: Array
    priority_index: Array
    full_index: Array


class CollectedAssignments(NamedTuple):
    """Keep whole action decisions and their original info ownership.

    trace has capacity 4B, with policy_ids shaped (4B,10). Remaining fields are
    int32 (4B,): original info ID, original action index, played episode length
    and evidence index. Unknown component IDs remain -1; nothing is clipped.
    """

    trace: PolicyTrace
    info_episode_id: Array
    decision_step: Array
    episode_length: Array
    config_index: Array


class CollectionErrors(NamedTuple):
    """Keep every reported failure and the first private collection diagnostic.

    lifecycle_error is bool (B,). episode_tracking_error and code are int32 (B,)
    bitwise accumulations. Private code bits are capacity=1, ownership=2,
    changed config=4, metric disagreement=8 and invalid row=16. They do not extend
    the public tracking schema. Remaining int32 scalars describe the first bad
    logical step/lane/episode and expected/observed values; -1 means unavailable.
    Errors remain meaningful even when no output family contains a row.
    """

    lifecycle_error: Array
    episode_tracking_error: Array
    code: Array
    step: Array
    lane: Array
    episode_id: Array
    expected: Array
    observed: Array


class CollectedBuffers(NamedTuple):
    """Own bounded recording payloads, excluding learner outputs and memory.

    evidence and completions have capacity R. Starts and metric tables have
    capacity R when enabled. Assignments have capacity 4B; replay has capacity
    4C for per-step replay capacity C. None omits an optional family entirely.
    """

    evidence: ConfigEvidence
    starts: CollectedStarts | None
    completions: CollectedCompletions
    priority: MetricValues | None
    full: MetricValues | None
    assignments: CollectedAssignments | None
    replay: ReplayPackets | None


class CollectedCounts(NamedTuple):
    """Select occupied prefixes with checked int32 scalars; absent families use 0."""

    evidence: Array
    starts: Array
    completions: Array
    priority: Array
    full: Array
    assignments: Array
    replay: Array


class CollectedBatch(NamedTuple):
    """Pass bounded numerical records, occupied counts and complete errors.

    The collector uses capacity-shaped buffers internally. Its host drain slices
    each family to its count before calling the writer. The writer must still
    validate all references, claims and joins before publishing any record.
    """

    buffers: CollectedBuffers
    counts: CollectedCounts
    errors: CollectionErrors
