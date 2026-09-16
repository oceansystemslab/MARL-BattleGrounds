"""Define optional numerical recording records and shared host error checks.

EpisodeStartRecords carries compact source claims from a future tracker to the
writer. It contains only numerical leaves, so callers may carry it through JAX
loops. Constructing a record does not verify its claim. The writer owns that
check. validate_recording_errors rejects reported failures before any consumer
publishes replay, assignment or completion data.
"""

from typing import NamedTuple

import numpy as np
from jax import Array

TRACKING_ERROR_DECLARATION = 1
TRACKING_ERROR_ACCOUNTING = 2
TRACKING_ERROR_OVERFLOW = 4


class EpisodeStartRecords(NamedTuple):
    """Describe first-transition source claims with fixed numerical shapes.

    Attributes
    ----------
    episode_id : Array
        Int32 shape L. Producing episode ID, unique within its recording pass.
    reset_generation : Array
        Int32 shape L. Nonnegative reset generation belonging to that start.
    source_table_id : Array
        Uint32 shape L+(8,). All eight big-endian words of the ordered source
        bank's SHA256. All zero means no source reference was declared.
    source_index : Array
        Int32 shape L. Source-bank row; -1 means unknown.
    spawn_locations : Array
        Int32 shape L. 0 retains both source banks; 1 exchanges both complete
        banks; -1 means unknown or ambiguous. This is not actor input.
    episode_start_stage : Array
        Int32 shape L. Stage at first transition; -1 means undeclared.
    source_known : Array
        Bool shape L. A source relationship was declared. The writer must still
        verify it. False requires source_index and spawn_locations to be -1.
    authored_start : Array
        Bool shape L. An exact authored start remains custom, even if its
        configuration has a known source relationship.
    valid : Array
        Bool shape L. True selects a real start record. False is padding.

    Notes
    -----
    L is (), (B,) or collected leading axes such as (T,B), matching EpisodeInfo.
    The constructor stores supplied leaves without validation or copying. Pass
    source-bank configurations separately to RunWriter.register_episodes. No
    coordinates, full configurations, sessions or learning values belong here.
    """

    episode_id: Array
    reset_generation: Array
    source_table_id: Array
    source_index: Array
    spawn_locations: Array
    episode_start_stage: Array
    source_known: Array
    authored_start: Array
    valid: Array


def validate_recording_errors(infos: object) -> None:
    """Reject any supplied lifecycle or tracking failure before host recording.

    Parameters
    ----------
    infos : object
        EpisodeInfo or an object exposing lifecycle_error and optionally
        episode_tracking_error. Lifecycle flags must be bool; tracking flags
        must be int32. Every element is checked, including padding and lanes
        with no start or completion. None means tracking was not supplied.

    Raises
    ------
    ValueError
        Any lifecycle flag is true or any tracking code is nonzero. Unknown
        nonzero tracking bits also fail. The message names affected indices.
    TypeError
        A supplied flag has the wrong dtype.

    Notes
    -----
    Host-only. Reading these small arrays may synchronize device work. This
    helper changes nothing and does not transfer configurations or metric
    vectors. It cannot check errors the caller did not supply.
    """
    for name, expected_dtype in (
        ("lifecycle_error", np.dtype(np.bool_)),
        ("episode_tracking_error", np.dtype(np.int32)),
    ):
        supplied = getattr(infos, name, None)
        if supplied is None:
            continue
        values = np.asarray(supplied)
        if values.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype.name}")
        invalid = values != 0
        if np.any(invalid):
            locations = [tuple(int(i) for i in row) for row in np.argwhere(invalid)]
            raise ValueError(
                f"recording rejected {name} at indices {locations}: "
                f"values {values[invalid].tolist()}"
            )
