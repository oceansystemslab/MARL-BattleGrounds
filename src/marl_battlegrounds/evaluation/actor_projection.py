"""Reconstruct recorded actor inputs without inventing extra information.

These host boundaries use immutable episode context and captured base rows.
Historical V2 class maps and V1 shared banks keep their declared projection;
current V3 context supplies current relative class rows and redacted V2 banks.
Numerical policy/JAX imports stay local until reconstruction is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV3,
    EvaluationFrameV1,
    EvaluationFrameV2,
    VersionedIdentityV1,
    evaluation_context_type,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1,
    MAX_AGENTS_PER_TEAM_V1,
    NUM_TEAMS_V1,
)

if TYPE_CHECKING:
    from marl_battlegrounds.policies.shared_obs import (
        SharedObsSensorSourceBankV1,
        SharedObsSensorSourceBankV2,
    )

NO_SHARED_OBS_ACTOR_PROJECTION_ID: Final = "base-observation-no-shared-obs"
NO_SHARED_OBS_ACTOR_PROJECTION_VERSION: Final = 3
NO_SHARED_OBS_ACTOR_PROJECTION_V2: Final = VersionedIdentityV1(
    identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID,
    version=2,
)
SHARED_OBS_ACTOR_PROJECTION_ID: Final = (
    "base-observation-plus-authorized-sensor-source-bank"
)
SHARED_OBS_ACTOR_PROJECTION_VERSION: Final = 2
SHARED_OBS_ACTOR_PROJECTION_V1: Final = VersionedIdentityV1(
    identifier=SHARED_OBS_ACTOR_PROJECTION_ID,
    version=1,
)

NO_SHARED_OBS_ACTOR_PROJECTION_V3: Final = VersionedIdentityV1(
    identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID, version=3
)
SHARED_OBS_ACTOR_PROJECTION_V2: Final = VersionedIdentityV1(
    identifier=SHARED_OBS_ACTOR_PROJECTION_ID, version=2
)

type ActorClassIdsByTeamV2 = tuple[tuple[int, ...], ...]
type ClassIdsByAgentByTeamV2 = tuple[ActorClassIdsByTeamV2, ...]


def _require_context(context: EvaluationEpisodeContext) -> None:
    """Require one of the exact immutable context versions supported by model
    readers.
    """
    evaluation_context_type(context)


def _derive_class_ids_by_agent_by_team(
    context: EvaluationEpisodeContext,
) -> ClassIdsByAgentByTeamV2:
    """Build public class rows using recorded actor-relative slot mappings.

    Return native Python tuples shaped (10, 2, 5): observer, ally/enemy team, row.
    Configured inactive observers get zero rows. No hidden state or global slot
    identifier is appended to the policy input.
    """
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
    """Require frozen Python tuples shaped (10, 2, 5) with exact int leaves."""
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
    context: EvaluationEpisodeContext,
    class_ids_by_agent_by_team: object,
) -> None:
    """Check a public class-ID observation against its recorded roster.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported immutable episode context.
    class_ids_by_agent_by_team : object
        Native Python tuple tree shaped (10, 2, 5),
        ordered observer, own/opposing team, actor-relative roster row.

    Returns
    -------
    None
        None when shape, exact integer types and every class row agree.

    Raises
    ------
    TypeError
        Context or a class ID has the wrong exact type.
    ValueError
        The tuple shape or any class assignment disagrees.

    Notes
    -----
    Host-only and read-only. Inactive observer rows must be zero. This checks
    the public roster map; it does not derive visibility or simulator state.
    """
    _require_class_id_payload_shape(class_ids_by_agent_by_team)
    expected = _derive_class_ids_by_agent_by_team(context)
    if class_ids_by_agent_by_team != expected:
        raise ValueError("observation class IDs do not match episode roster context")


def _require_no_shared_obs_projection_v2(
    context: EvaluationEpisodeContext,
) -> None:
    """Require the historical NoSharedObs V2 projection identity and mode."""
    _require_context(context)
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError("actor projection V2 requires no_shared_obs execution")
    if context.actor_projection != NO_SHARED_OBS_ACTOR_PROJECTION_V2:
        raise ValueError(
            "actor projection V2 requires base-observation-no-shared-obs version 2"
        )


def _require_shared_obs_projection_v1(
    context: EvaluationEpisodeContext,
) -> None:
    """Require the historical SharedObs V1 projection identity and mode."""
    _require_context(context)
    if context.execution_information_mode != "shared_obs":
        raise ValueError("SharedObs projection V1 requires shared_obs execution")
    if context.actor_projection != SHARED_OBS_ACTOR_PROJECTION_V1:
        raise ValueError(
            "SharedObs projection V1 requires "
            "base-observation-plus-authorized-sensor-source-bank version 1"
        )


def reconstruct_shared_obs_sensor_source_bank_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV1,
) -> SharedObsSensorSourceBankV1:
    """Rebuild the historical two-team sensor bank from recorded base rows.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Immutable context declaring the SharedObs V1 projection.
    frame : EvaluationFrameV1
        Exact EvaluationFrameV1 for that episode, including recorded source
        availability and base observation rows.

    Returns
    -------
    SharedObsSensorSourceBankV1
        SharedObsSensorSourceBankV1 with JAX array leaves. The leading source layout
        follows both teams' recorded actor mappings. This is the shared bank before
        per-recipient redaction; callers must enforce each recipient's permissions.

    Raises
    ------
    TypeError
        Context/frame has an unsupported exact type.
    ValueError
        Projection/mode, episode join or recorded availability is wrong.

    Notes
    -----
    Host-only reconstruction with local numerical imports; it may create device
    arrays. It reads captured rows and fixed mappings rather than rerunning
    visibility or persisting duplicated actor inputs. Inputs remain unchanged.
    """
    _require_shared_obs_projection_v1(context)
    if type(frame) is not EvaluationFrameV1:
        raise TypeError(
            "SharedObs source-bank reconstruction requires EvaluationFrameV1"
        )
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("SharedObs frame and context episode IDs must match")
    if frame.shared_obs_information_availability_by_recipient_and_sensor_source is None:
        raise ValueError("SharedObs frame requires recorded information availability")

    # Imported only at the execution reconstruction boundary.
    import jax.numpy as jnp

    from marl_battlegrounds.core.types import (
        AGENT_FEATURE_ACTIVE,
        AGENT_FEATURE_ALIVE,
    )
    from marl_battlegrounds.policies.shared_obs import (
        build_shared_obs_sensor_source_bank_from_base_rows,
    )

    base = frame.base_observation
    catalog = context.static_mechanics_catalog
    self_features = jnp.asarray(base.self_features, dtype=jnp.float32)
    source_is_living = jnp.logical_and(
        self_features[:, AGENT_FEATURE_ACTIVE] > 0.0,
        self_features[:, AGENT_FEATURE_ALIVE] > 0.0,
    )
    return build_shared_obs_sensor_source_bank_from_base_rows(
        jnp.asarray(base.ally_unit_features, dtype=jnp.float32),
        jnp.asarray(base.enemy_unit_features, dtype=jnp.float32),
        jnp.asarray(base.objective_features, dtype=jnp.float32),
        jnp.asarray(base.ally_visibility_mask, dtype=jnp.bool_),
        jnp.asarray(base.enemy_visibility_mask, dtype=jnp.bool_),
        source_is_living,
        global_slot_by_actor_and_ally_observation_row=jnp.asarray(
            catalog.global_slot_by_actor_and_ally_observation_row,
            dtype=jnp.int32,
        ),
        global_slot_by_actor_and_enemy_observation_row=jnp.asarray(
            catalog.global_slot_by_actor_and_enemy_observation_row,
            dtype=jnp.int32,
        ),
    )


def reconstruct_class_ids_by_agent_by_team_v2(
    context: EvaluationEpisodeContext,
) -> ClassIdsByAgentByTeamV2:
    """Rebuild all historical NoSharedObs V2 public roster rows.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Immutable context declaring the NoSharedObs V2 projection.

    Returns
    -------
    ClassIdsByAgentByTeamV2
        Python int tuples shaped (10, 2, 5): observer, own/opposing team, roster row.
        Inactive observers have zeros.

    Raises
    ------
    TypeError
        Context is not an exact supported model.
    ValueError
        Information mode or projection identity is incompatible.

    Notes
    -----
    Host-only, deterministic and read-only. Global slots are used only for
    reconstruction; they are not added to an actor's public row.
    """
    _require_no_shared_obs_projection_v2(context)
    return _derive_class_ids_by_agent_by_team(context)


def reconstruct_actor_class_ids_by_team_v2(
    context: EvaluationEpisodeContext,
    global_slot: int,
) -> ActorClassIdsByTeamV2:
    """Rebuild one historical NoSharedObs V2 actor's public roster rows.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Immutable context declaring the NoSharedObs V2 projection.
    global_slot : int
        Exact Python int in 0..9 selecting the observer internally.

    Returns
    -------
    ActorClassIdsByTeamV2
        Python int tuples shaped (2, 5), own team first and opponent team second.
        An inactive observer receives zero rows.

    Raises
    ------
    TypeError
        Context is not an exact supported model.
    ValueError
        Projection/mode or global_slot is invalid.

    Notes
    -----
    Host-only. The selector is not returned as extra policy information.
    """
    _require_no_shared_obs_projection_v2(context)
    if type(global_slot) is not int or not 0 <= global_slot < MAX_AGENT_SLOTS_V1:
        raise ValueError("global_slot must be an exact integer in [0, 10)")
    return _derive_class_ids_by_agent_by_team(context)[global_slot]


__all__ = (
    "NO_SHARED_OBS_ACTOR_PROJECTION_ID",
    "NO_SHARED_OBS_ACTOR_PROJECTION_V2",
    "NO_SHARED_OBS_ACTOR_PROJECTION_V3",
    "NO_SHARED_OBS_ACTOR_PROJECTION_VERSION",
    "SHARED_OBS_ACTOR_PROJECTION_ID",
    "SHARED_OBS_ACTOR_PROJECTION_V1",
    "SHARED_OBS_ACTOR_PROJECTION_V2",
    "SHARED_OBS_ACTOR_PROJECTION_VERSION",
    "reconstruct_actor_class_ids_by_team_v2",
    "reconstruct_actor_class_ids_by_team_v3",
    "reconstruct_class_ids_by_agent_by_team_v2",
    "reconstruct_class_ids_by_agent_by_team_v3",
    "reconstruct_shared_obs_sensor_source_bank_v1",
    "reconstruct_shared_obs_sensor_source_bank_v2",
    "validate_class_ids_by_agent_by_team_against_context_v1",
)


def _require_current_projection(context: EvaluationEpisodeContext) -> None:
    """Require exact context V3 and the current projection matching its information
    mode.
    """
    if type(context) is not EvaluationEpisodeContextV3:
        raise TypeError("current actor reconstruction requires context V3")
    expected = (
        SHARED_OBS_ACTOR_PROJECTION_V2
        if context.execution_information_mode == "shared_obs"
        else NO_SHARED_OBS_ACTOR_PROJECTION_V3
    )
    if context.actor_projection != expected:
        raise ValueError("current actor projection does not match its information mode")


def reconstruct_class_ids_by_agent_by_team_v3(
    context: EvaluationEpisodeContext,
) -> ClassIdsByAgentByTeamV2:
    """Rebuild the current public class map from exact V3 context.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        EvaluationEpisodeContextV3 with its matching current actor
        projection, either SharedObs V2 or NoSharedObs V3.

    Returns
    -------
    ClassIdsByAgentByTeamV2
        Python int tuples shaped (10, 2, 5): observer, own/opposing team, row.
        Inactive observer rows are zero.

    Raises
    ------
    TypeError
        Context is not exact V3.
    ValueError
        The projection does not match the declared information mode.

    Notes
    -----
    Host-only and read-only. This public roster data is static context; no
    unseen positions, visibility or privileged dynamics are reconstructed.
    """
    _require_current_projection(context)
    return _derive_class_ids_by_agent_by_team(context)


def reconstruct_actor_class_ids_by_team_v3(
    context: EvaluationEpisodeContext, global_slot: int
) -> ActorClassIdsByTeamV2:
    """Rebuild one current actor's public class rows from exact V3 context.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        EvaluationEpisodeContextV3 with the matching current projection.
    global_slot : int
        Exact Python int in 0..9 selecting the observer internally.

    Returns
    -------
    ActorClassIdsByTeamV2
        Python int tuples shaped (2, 5), ordered own team then opponent team.
        Inactive observers receive zero rows.

    Raises
    ------
    TypeError
        Context is not exact V3.
    ValueError
        The projection or slot is invalid.

    Notes
    -----
    Host-only and read-only. The global selector never becomes an added actor
    input; it only locates that actor's recorded relative rows.
    """
    _require_current_projection(context)
    if type(global_slot) is not int or not 0 <= global_slot < MAX_AGENT_SLOTS_V1:
        raise ValueError("global_slot must be an exact integer in [0, 10)")
    return _derive_class_ids_by_agent_by_team(context)[global_slot]


def reconstruct_shared_obs_sensor_source_bank_v2(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV2,
    selected_global_slot: int,
) -> SharedObsSensorSourceBankV2:
    """Restore one current actor's permitted five-source sensor bank.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact EvaluationEpisodeContextV3 declaring SharedObs V2.
    frame : EvaluationFrameV2
        Exact EvaluationFrameV2 for the same episode, with a recorded
        boolean (10, 10) recipient/source availability matrix.
    selected_global_slot : int
        Exact Python int in 0..9 selecting the recipient.

    Returns
    -------
    SharedObsSensorSourceBankV2
        SharedObsSensorSourceBankV2 with five team-source rows in the recipient's
        recorded ally order. Float data is float32, visibility/admission is boolean;
        unavailable source rows are masked by the policy-layer authority.

    Raises
    ------
    TypeError
        Context is not exact V3.
    ValueError
        Projection, mode, frame join, slot or availability is invalid,
        including self, inactive or opposing sources declared as admitted.

    Notes
    -----
    Host-only reconstruction creates JAX arrays and may transfer to the current
    device. It uses captured source visibility and recorded mappings, then
    redacts for this actor. It does not run game physics or edit the recording.
    """
    _require_current_projection(context)
    if context.execution_information_mode != "shared_obs":
        raise ValueError("SharedObs reconstruction requires shared_obs execution")
    if (
        type(frame) is not EvaluationFrameV2
        or frame.episode_id != context.identity.episode_id
    ):
        raise ValueError("current source-bank frame must be V2 and join its context")
    if (
        type(selected_global_slot) is not int
        or not 0 <= selected_global_slot < MAX_AGENT_SLOTS_V1
    ):
        raise ValueError("selected_global_slot must be an exact integer in [0, 10)")
    availability = (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source
    )
    if availability is None:
        raise ValueError("SharedObs reconstruction requires recorded availability")
    recipient = context.roster[selected_global_slot]
    for source_slot, admitted in enumerate(availability[selected_global_slot]):
        source = context.roster[source_slot]
        if admitted and (
            source_slot == selected_global_slot
            or not recipient.configured_active
            or not source.configured_active
            or recipient.configured_team_id != source.configured_team_id
        ):
            raise ValueError("recorded source availability includes a forbidden source")
    import jax.numpy as jnp

    from marl_battlegrounds.policies.shared_obs import (
        build_shared_obs_team_source_bank_from_base_rows,
        mask_source_bank_for_recipient,
    )

    rows = (
        context.static_mechanics_catalog.global_slot_by_actor_and_ally_observation_row[
            selected_global_slot
        ]
    )
    base = frame.base_observation

    self_rows = jnp.asarray(
        tuple(base.self_features[index] for index in rows), dtype=jnp.float32
    )
    bank = build_shared_obs_team_source_bank_from_base_rows(
        jnp.asarray(
            tuple(base.ally_unit_features[index] for index in rows), dtype=jnp.float32
        ),
        jnp.asarray(
            tuple(base.enemy_unit_features[index] for index in rows), dtype=jnp.float32
        ),
        jnp.asarray(
            tuple(base.objective_features[index] for index in rows), dtype=jnp.float32
        ),
        jnp.asarray(
            tuple(base.ally_visibility_mask[index] for index in rows), dtype=jnp.bool_
        ),
        jnp.asarray(
            tuple(base.enemy_visibility_mask[index] for index in rows), dtype=jnp.bool_
        ),
        (self_rows[:, 4] == 1.0) & (self_rows[:, 5] == 1.0),
    )
    admitted = jnp.asarray(
        tuple(availability[selected_global_slot][index] for index in rows),
        dtype=jnp.bool_,
    )
    return mask_source_bank_for_recipient(bank, admitted)
