"""Host-only actor-input reconstruction from immutable evaluation context."""

from typing import Final, cast

from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    VersionedIdentityV1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1,
    MAX_AGENTS_PER_TEAM_V1,
    NUM_TEAMS_V1,
)

NO_SHARED_OBS_ACTOR_PROJECTION_ID: Final = "base-observation-no-shared-obs"
NO_SHARED_OBS_ACTOR_PROJECTION_VERSION: Final = 2
NO_SHARED_OBS_ACTOR_PROJECTION_V2: Final = VersionedIdentityV1(
    identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID,
    version=NO_SHARED_OBS_ACTOR_PROJECTION_VERSION,
)

type ActorClassIdsByTeamV2 = tuple[tuple[int, ...], ...]
type ClassIdsByAgentByTeamV2 = tuple[ActorClassIdsByTeamV2, ...]


def _require_context(context: EvaluationEpisodeContextV1) -> None:
    """Require the exact immutable context model used by this projection."""
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("actor projection requires EvaluationEpisodeContextV1")


def _derive_class_ids_by_agent_by_team(
    context: EvaluationEpisodeContextV1,
) -> ClassIdsByAgentByTeamV2:
    """Derive actor-relative class rows from serialized roster/mapping authority."""
    _require_context(context)
    catalog = context.static_mechanics_catalog
    roster = context.roster
    zero_team = (0,) * MAX_AGENTS_PER_TEAM_V1
    rows: list[ActorClassIdsByTeamV2] = []

    for global_slot, observer in enumerate(roster):
        if not observer.configured_active:
            rows.append((zero_team,) * NUM_TEAMS_V1)
            continue
        ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[global_slot]
        enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[
            global_slot
        ]
        rows.append(
            (
                tuple(roster[slot].class_id for slot in ally_slots),
                tuple(roster[slot].class_id for slot in enemy_slots),
            )
        )

    return tuple(rows)


def _require_class_id_payload_shape(class_ids_by_agent_by_team: object) -> None:
    """Require one frozen native-Python ``(10, 2, 5)`` integer payload."""
    if type(class_ids_by_agent_by_team) is not tuple:
        raise ValueError("class IDs must have shape (10, 2, 5)")
    observer_payload = cast(tuple[object, ...], class_ids_by_agent_by_team)
    if len(observer_payload) != MAX_AGENT_SLOTS_V1:
        raise ValueError("class IDs must have shape (10, 2, 5)")
    for observer_value in observer_payload:
        if type(observer_value) is not tuple:
            raise ValueError("class IDs must have shape (10, 2, 5)")
        observer_rows = cast(tuple[object, ...], observer_value)
        if len(observer_rows) != NUM_TEAMS_V1:
            raise ValueError("class IDs must have shape (10, 2, 5)")
        for team_value in observer_rows:
            if type(team_value) is not tuple:
                raise ValueError("class IDs must have shape (10, 2, 5)")
            team_row = cast(tuple[object, ...], team_value)
            if len(team_row) != MAX_AGENTS_PER_TEAM_V1:
                raise ValueError("class IDs must have shape (10, 2, 5)")
            if any(type(class_id) is not int for class_id in team_row):
                raise TypeError("class IDs must contain exact integers")


def validate_class_ids_by_agent_by_team_against_context_v1(
    context: EvaluationEpisodeContextV1,
    class_ids_by_agent_by_team: object,
) -> None:
    """Fail when a live public class map disagrees with immutable V1 context."""
    _require_class_id_payload_shape(class_ids_by_agent_by_team)
    expected = _derive_class_ids_by_agent_by_team(context)
    if class_ids_by_agent_by_team != expected:
        raise ValueError("observation class IDs do not match episode roster context")


def _require_no_shared_obs_projection_v2(
    context: EvaluationEpisodeContextV1,
) -> None:
    """Require the exact supported information regime and projection identity."""
    _require_context(context)
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError("actor projection V2 requires no_shared_obs execution")
    if context.actor_projection != NO_SHARED_OBS_ACTOR_PROJECTION_V2:
        raise ValueError(
            "actor projection V2 requires base-observation-no-shared-obs version 2"
        )


def reconstruct_class_ids_by_agent_by_team_v2(
    context: EvaluationEpisodeContextV1,
) -> ClassIdsByAgentByTeamV2:
    """Reconstruct the complete public ``(10, 2, 5)`` class-ID observation leaf."""
    _require_no_shared_obs_projection_v2(context)
    return _derive_class_ids_by_agent_by_team(context)


def reconstruct_actor_class_ids_by_team_v2(
    context: EvaluationEpisodeContextV1,
    global_slot: int,
) -> ActorClassIdsByTeamV2:
    """Reconstruct one actor's public ``(2, 5)`` class-ID observation row."""
    _require_no_shared_obs_projection_v2(context)
    if type(global_slot) is not int or not 0 <= global_slot < MAX_AGENT_SLOTS_V1:
        raise ValueError("global_slot must be an exact integer in [0, 10)")
    return _derive_class_ids_by_agent_by_team(context)[global_slot]


__all__ = (
    "NO_SHARED_OBS_ACTOR_PROJECTION_ID",
    "NO_SHARED_OBS_ACTOR_PROJECTION_V2",
    "NO_SHARED_OBS_ACTOR_PROJECTION_VERSION",
    "reconstruct_actor_class_ids_by_team_v2",
    "reconstruct_class_ids_by_agent_by_team_v2",
    "validate_class_ids_by_agent_by_team_against_context_v1",
)
