"""Copy simulator outputs into durable host records and restore recorded arrays.

Capture transfers the complete input bundle with one jax.device_get call, then
checks exact NumPy shapes/dtypes before building strict models. This is a host
recording boundary, outside JIT; one bundled call is not a claim about physical
transfer count. Reconstruction reads recorded values without running physics.
Current context V4 uses frame V3 (20 context columns); context V3 uses frame V2
and older contexts keep frame V1.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike, NDArray

from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    SELF_FEATURES,
    UNIT_FEATURES,
    Action,
    ActionAcceptanceFacts,
    ActionMask,
    AuraTransitionFacts,
    CombatTransitionFacts,
    DeathTransitionFacts,
    DoneFlags,
    EnvState,
    Observation,
    PhysicalTransitionFacts,
    PreviousTimestepActionObservation,
    RegenerationTransitionFacts,
    RespawnTransitionFacts,
    Reward,
    SpawnLifecycleObservation,
    SpawnShieldTransitionFacts,
    StatusLifecycleTransitionFacts,
    TeamDeathmatchTransitionFacts,
    TransitionFacts,
)
from marl_battlegrounds.evaluation.actor_projection import (
    validate_class_ids_by_agent_by_team_against_context_v1,
)
from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    ActionMaskV1,
    AuraTransitionFactsV1,
    BaseObservationV1,
    BaseObservationV2,
    BaseObservationV3,
    CombatTransitionFactsV1,
    DeathTransitionFactsV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeContextV3,
    EvaluationEpisodeContextV4,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationFrameV2,
    EvaluationFrameV3,
    EvaluationTransitionV1,
    GlobalAnalysisSnapshotV1,
    JointActionV1,
    PhysicalTransitionFactsV1,
    PreviousTimestepActionObservationV1,
    RegenerationTransitionFactsV1,
    RespawnTransitionFactsV1,
    SpawnLifecycleObservationV1,
    SpawnShieldTransitionFactsV1,
    StatusLifecycleTransitionFactsV1,
    TeamDeathmatchTransitionFactsV1,
    TransitionFactsV1,
    evaluation_context_type,
    evaluation_frame_type,
    evaluation_frame_type_for_context,
)
from marl_battlegrounds.evaluation.validation import (
    _derive_and_validate_team_deathmatch_authority_v1,  # pyright: ignore[reportPrivateUsage]
    _validate_frame_information_regime,  # pyright: ignore[reportPrivateUsage]
    validate_evaluation_transition_unit_v1,
    validate_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    CONTEXT_FEATURES_V1,
    CONTEXT_FEATURES_V2,
)

_BOOL_DTYPE = np.dtype(np.bool_)
_INT32_DTYPE = np.dtype(np.int32)
_FLOAT32_DTYPE = np.dtype(np.float32)
_NUM_STATUS_CHANNELS = NUM_SLOW_CHANNELS + NUM_STUN_CHANNELS + 3


def _require_exact_type(
    value: object,
    expected_type: type[object],
    *,
    name: str,
) -> None:
    """Reject a source whose concrete type differs from the required Core record.

    Raise TypeError with the supplied field name; subclasses are not accepted.
    """
    if type(value) is not expected_type:
        raise TypeError(
            f"{name} must be exactly {expected_type.__name__}, "
            f"not {type(value).__name__}"
        )


def _require_host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    finite: bool = False,
    category_count: int | None = None,
) -> NDArray[np.generic]:
    """Check an already-transferred NumPy leaf without converting or copying it.

    Require exact ndarray type, declared shape and dtype. Optional finite=False
    and category_count=None skip those checks; a category bound means integers from
    zero inclusive to that bound exclusive. Return the same array. Reject JAX leaves
    and wrong types with TypeError, and invalid values/shapes with ValueError.
    """
    if isinstance(value, jax.Array):
        raise TypeError(
            f"{name} is still a JAX/device array; transfer the complete source "
            "bundle before normalization"
        )
    if type(value) is not np.ndarray:
        raise TypeError(
            f"{name} must be an exact NumPy array after device_get, "
            f"not {type(value).__name__}"
        )
    array = cast(NDArray[np.generic], value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, not {array.dtype}")
    if finite and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    if category_count is not None and bool(
        np.any(
            (cast(NDArray[np.int32], array) < 0)
            | (cast(NDArray[np.int32], array) >= category_count)
        )
    ):
        raise ValueError(f"{name} contains an out-of-domain category")
    return array


def _freeze_payload(value: object) -> object:
    """Convert nested lists from ndarray.tolist into tuples and native scalars.

    Accept only exact bool, int, and float leaves. Raise TypeError for any other
    scalar; this helper never accepts or transfers device arrays.
    """
    if isinstance(value, list):
        return tuple(_freeze_payload(item) for item in cast(list[object], value))
    if type(value) in (bool, int, float):
        return value
    raise TypeError(f"unsupported host payload scalar {type(value).__name__}")


def _array_payload(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    finite: bool = False,
    category_count: int | None = None,
) -> object:
    """Validate one host leaf and return its immutable Python scalar/tuple payload.

    Use the declared exact shape/dtype plus optional finite/category checks.
    Delegates errors to _require_host_array and performs no device transfer.
    """
    array = _require_host_array(
        value,
        name=name,
        shape=shape,
        dtype=dtype,
        finite=finite,
        category_count=category_count,
    )
    return _freeze_payload(cast(object, array.tolist()))


def _normalize_snapshot_v1(state: EnvState) -> tuple[int, GlobalAnalysisSnapshotV1]:
    """Convert one unbatched host EnvState into simulator tick and V1 snapshot.

    Require exact Core type, fixed global/team axes, int32 counters/actions, float32
    positions/health, and bool flags. Reject negative tick, nonfinite floats, or
    out-of-range previous actions. Values remain recorded snapshot facts.
    """
    _require_exact_type(state, EnvState, name="state")
    step_count = cast(
        int,
        _array_payload(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=_INT32_DTYPE,
        ),
    )
    if step_count < 0:
        raise ValueError("state.step_count must be nonnegative")

    payload: dict[str, object] = {
        "team_deathmatch_scores": _array_payload(
            state.team_deathmatch_scores,
            name="state.team_deathmatch_scores",
            shape=(NUM_TEAMS,),
            dtype=_INT32_DTYPE,
        ),
        "alive_mask": _array_payload(
            state.alive_mask,
            name="state.alive_mask",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_BOOL_DTYPE,
        ),
        "agent_positions": _array_payload(
            state.agent_positions,
            name="state.agent_positions",
            shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
            dtype=_FLOAT32_DTYPE,
            finite=True,
        ),
        "current_health": _array_payload(
            state.current_health,
            name="state.current_health",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_FLOAT32_DTYPE,
            finite=True,
        ),
        "ultimate_cooldowns": _array_payload(
            state.ultimate_cooldowns,
            name="state.ultimate_cooldowns",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "slow_durations": _array_payload(
            state.slow_durations,
            name="state.slow_durations",
            shape=(MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS),
            dtype=_INT32_DTYPE,
        ),
        "stun_durations": _array_payload(
            state.stun_durations,
            name="state.stun_durations",
            shape=(MAX_AGENT_SLOTS, NUM_STUN_CHANNELS),
            dtype=_INT32_DTYPE,
        ),
        "rogue_poison_anti_heal_durations": _array_payload(
            state.rogue_poison_anti_heal_durations,
            name="state.rogue_poison_anti_heal_durations",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "mage_burst_damage_amplification_durations": _array_payload(
            state.mage_burst_damage_amplification_durations,
            name="state.mage_burst_damage_amplification_durations",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "priest_blessing_of_freedom_slow_floor_durations": _array_payload(
            state.priest_blessing_of_freedom_slow_floor_durations,
            name="state.priest_blessing_of_freedom_slow_floor_durations",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "team_respawn_wave_countdowns": _array_payload(
            state.team_respawn_wave_countdowns,
            name="state.team_respawn_wave_countdowns",
            shape=(NUM_TEAMS,),
            dtype=_INT32_DTYPE,
        ),
        "spawn_shield_durations": _array_payload(
            state.spawn_shield_durations,
            name="state.spawn_shield_durations",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "steps_until_out_of_combat": _array_payload(
            state.steps_until_out_of_combat,
            name="state.steps_until_out_of_combat",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
        ),
        "previous_timestep_move_actions": _array_payload(
            state.previous_timestep_move_actions,
            name="state.previous_timestep_move_actions",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
            category_count=NUM_MOVE_ACTIONS,
        ),
        "previous_timestep_select_target_actions": _array_payload(
            state.previous_timestep_select_target_actions,
            name="state.previous_timestep_select_target_actions",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
            category_count=NUM_TARGET_ACTIONS,
        ),
        "previous_timestep_use_ultimate_actions": _array_payload(
            state.previous_timestep_use_ultimate_actions,
            name="state.previous_timestep_use_ultimate_actions",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
            category_count=NUM_ULTIMATE_ACTIONS,
        ),
        "has_previous_timestep_joint_action": _array_payload(
            state.has_previous_timestep_joint_action,
            name="state.has_previous_timestep_joint_action",
            shape=(),
            dtype=_BOOL_DTYPE,
        ),
    }
    return step_count, GlobalAnalysisSnapshotV1.model_validate(payload)


def _normalize_previous_action_observation_v1(
    source: PreviousTimestepActionObservation,
) -> PreviousTimestepActionObservationV1:
    """Freeze the six ally/enemy previous-action tensors for ten observers.

    Require finite float32 arrays shaped (10, 5, head_categories), preserving actor
    and relation-row order. Producers supply one-hot or neutral rows; strict wire
    models check finite values and shapes.
    The Core source must already contain NumPy leaves.
    """
    _require_exact_type(
        source,
        PreviousTimestepActionObservation,
        name="observation.previous_timestep_actions",
    )
    fields_and_shapes = (
        (
            "ally_previous_timestep_move_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_MOVE_ACTIONS),
        ),
        (
            "enemy_previous_timestep_move_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_MOVE_ACTIONS),
        ),
        (
            "ally_previous_timestep_select_target_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
        ),
        (
            "enemy_previous_timestep_select_target_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
        ),
        (
            "ally_previous_timestep_use_ultimate_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_ULTIMATE_ACTIONS),
        ),
        (
            "enemy_previous_timestep_use_ultimate_actions_one_hot",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_ULTIMATE_ACTIONS),
        ),
    )
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"observation.previous_timestep_actions.{field_name}",
            shape=shape,
            dtype=_FLOAT32_DTYPE,
            finite=True,
        )
        for field_name, shape in fields_and_shapes
    }
    return PreviousTimestepActionObservationV1.model_validate(payload)


def _normalize_spawn_lifecycle_observation_v1(
    source: SpawnLifecycleObservation,
    context: EvaluationEpisodeContext,
) -> SpawnLifecycleObservationV1:
    """Freeze host spawn observations after checking class rows against the roster.

    Require fixed (observer, own/opponent team, local slot) axes, world-unit pad
    positions, int32 tick fields, float32 speed, and bool masks. The validated class
    map is omitted from the historical V1 wire subtree and reconstructed from
    context by the actor projection; it is not silently dropped without checking.
    """
    _require_exact_type(
        source,
        SpawnLifecycleObservation,
        name="observation.spawn_lifecycle",
    )
    class_ids = _array_payload(
        source.class_ids_by_agent_by_team,
        name="observation.spawn_lifecycle.class_ids_by_agent_by_team",
        shape=(MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM),
        dtype=_INT32_DTYPE,
    )
    validate_class_ids_by_agent_by_team_against_context_v1(context, class_ids)

    specs = (
        (
            "spawn_pad_positions_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "spawn_shield_actual_durations_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM),
            _INT32_DTYPE,
            False,
        ),
        (
            "spawn_shield_configured_duration_by_agent",
            (MAX_AGENT_SLOTS,),
            _INT32_DTYPE,
            False,
        ),
        (
            "spawn_shield_speed_by_agent",
            (MAX_AGENT_SLOTS,),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "respawn_wave_period_step_count_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS),
            _INT32_DTYPE,
            False,
        ),
        (
            "respawn_wave_countdowns_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS),
            _INT32_DTYPE,
            False,
        ),
        (
            "active_mask_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM),
            _BOOL_DTYPE,
            False,
        ),
        (
            "alive_mask_by_agent_by_team",
            (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM),
            _BOOL_DTYPE,
            False,
        ),
    )
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"observation.spawn_lifecycle.{field_name}",
            shape=shape,
            dtype=dtype,
            finite=finite,
        )
        for field_name, shape, dtype, finite in specs
    }
    # V1 bytes remain immutable; the validated class map is reconstructed from
    # the episode roster by actor projection V2 rather than serialized here.
    return SpawnLifecycleObservationV1.model_validate(payload)


def _normalize_base_observation(
    source: Observation,
    context: EvaluationEpisodeContext,
) -> BaseObservationV1 | BaseObservationV2 | BaseObservationV3:
    """Freeze one host observation using the context's declared wire version.

    Check exact shapes, bool masks, finite float32 features, previous-action rows,
    and spawn data. Context V4 records 20 context columns (column 19 is the Red
    Zone depth) and int32 self_ally_index in BaseObservationV3; context V3 the
    same with 19 columns in BaseObservationV2. The context width comes from the
    recorded wire version, not from the live simulator, so a live 20-column
    observation cannot be captured under a historical context. Legacy capture
    requires historical Team ID self features and returns V1; current
    identity-blind observations cannot be relabeled as legacy data.
    """
    _require_exact_type(source, Observation, name="observation")
    context_width = (
        CONTEXT_FEATURES_V2
        if type(context) is EvaluationEpisodeContextV4
        else CONTEXT_FEATURES_V1
    )
    specs = (
        ("self_features", (MAX_AGENT_SLOTS, SELF_FEATURES), _FLOAT32_DTYPE, True),
        (
            "ally_unit_features",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "enemy_unit_features",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "map_obstacle_features",
            (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "objective_features",
            (MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "context_features",
            (MAX_AGENT_SLOTS, context_width),
            _FLOAT32_DTYPE,
            True,
        ),
        (
            "ally_visibility_mask",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM),
            _BOOL_DTYPE,
            False,
        ),
        (
            "enemy_visibility_mask",
            (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM),
            _BOOL_DTYPE,
            False,
        ),
    )
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"observation.{field_name}",
            shape=shape,
            dtype=dtype,
            finite=finite,
        )
        for field_name, shape, dtype, finite in specs
    }
    payload["previous_timestep_actions"] = _normalize_previous_action_observation_v1(
        source.previous_timestep_actions
    )
    payload["spawn_lifecycle"] = _normalize_spawn_lifecycle_observation_v1(
        source.spawn_lifecycle,
        context,
    )
    if type(context) in (EvaluationEpisodeContextV3, EvaluationEpisodeContextV4):
        payload["self_ally_index"] = _array_payload(
            source.self_ally_index,
            name="observation.self_ally_index",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
            category_count=MAX_AGENTS_PER_TEAM,
        )
        if type(context) is EvaluationEpisodeContextV4:
            return BaseObservationV3.model_validate(payload)
        return BaseObservationV2.model_validate(payload)
    for slot, row in enumerate(context.roster):
        if source.self_features[slot, 3] != float(row.configured_team_id):
            raise ValueError(
                "legacy capture requires historical Team ID observations; "
                "use current capture V2"
            )
    return BaseObservationV1.model_validate(payload)


def _normalize_action_mask_v1(source: ActionMask) -> ActionMaskV1:
    """Freeze host boolean masks with fixed actor/head axes.

    Require shapes (10, 9), (10, 11), (10, 2), and (10, 11, 2) for movement,
    target, Ultimate, and their joint combat mask. No legality is recomputed.
    """
    _require_exact_type(source, ActionMask, name="action_mask")
    specs = (
        ("move_mask", (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)),
        ("select_target_mask", (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)),
        ("use_ultimate_mask", (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)),
        (
            "select_target_use_ultimate_joint_mask",
            (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
        ),
    )
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"action_mask.{field_name}",
            shape=shape,
            dtype=_BOOL_DTYPE,
        )
        for field_name, shape in specs
    }
    return ActionMaskV1.model_validate(payload)


def _normalize_joint_action_v1(
    source: Action,
    *,
    name: str,
    require_accepted_domains: bool,
) -> JointActionV1:
    """Freeze three int32 action vectors while preserving submitted invalid intent.

    Require shape (10,) for every head. When require_accepted_domains is true,
    enforce each head's category range; otherwise retain any int32 submission for
    rejection analysis. The supplied name identifies errors.
    """
    _require_exact_type(source, Action, name=name)
    categories = (
        ("move", NUM_MOVE_ACTIONS),
        ("select_target", NUM_TARGET_ACTIONS),
        ("use_ultimate", NUM_ULTIMATE_ACTIONS),
    )
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"{name}.{field_name}",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_INT32_DTYPE,
            category_count=(category_count if require_accepted_domains else None),
        )
        for field_name, category_count in categories
    }
    return JointActionV1.model_validate(payload)


def _normalize_action_acceptance_facts_v1(
    source: ActionAcceptanceFacts,
) -> ActionAcceptanceFactsV1:
    """Freeze submitted/accepted actions and three independent rejection-flag rows.

    Only accepted actions require category bounds. All rejection flags are bool
    vectors of length ten. Source leaves must already be on the host.
    """
    _require_exact_type(source, ActionAcceptanceFacts, name="action_acceptance_facts")
    payload: dict[str, object] = {
        "submitted_joint_action": _normalize_joint_action_v1(
            source.submitted_joint_action,
            name="action_acceptance_facts.submitted_joint_action",
            require_accepted_domains=False,
        ),
        "accepted_joint_action": _normalize_joint_action_v1(
            source.accepted_joint_action,
            name="action_acceptance_facts.accepted_joint_action",
            require_accepted_domains=True,
        ),
    }
    for field_name in (
        "submitted_action_tuple_is_out_of_domain_by_actor",
        "in_domain_move_action_is_rejected_by_actor",
        "in_domain_combat_action_pair_is_rejected_by_actor",
    ):
        payload[field_name] = _array_payload(
            getattr(source, field_name),
            name=f"action_acceptance_facts.{field_name}",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_BOOL_DTYPE,
        )
    return ActionAcceptanceFactsV1.model_validate(payload)


def _normalize_combat_transition_facts_v1(
    source: CombatTransitionFacts,
) -> CombatTransitionFactsV1:
    """Freeze combat facts and translate absent recipient routes from -1 to None.

    Require fixed source/recipient axes, finite float32 amounts/modifiers, and bool
    application flags. A present recipient must be a global slot 0-9; an absent
    recipient must carry Core's exact -1 sentinel. Invalid route joins raise ValueError.
    """
    _require_exact_type(source, CombatTransitionFacts, name="combat_transition_facts")
    bool_vectors = (
        "basic_effect_is_activated_by_source",
        "ultimate_effect_is_activated_by_source",
        "combat_effect_has_recipient_by_source",
        "rogue_poison_anti_heal_is_applied_by_source",
        "mage_burst_damage_amplification_is_applied_by_source",
        "priest_blessing_of_freedom_is_applied_by_source",
    )
    float_vectors = (
        "raw_damage_output_by_source",
        "source_modified_damage_output_by_source",
        "recipient_damage_modifier_by_source",
        "total_effective_damage_by_recipient",
        "raw_healing_output_by_source",
        "source_modified_healing_output_by_source",
        "recipient_healing_modifier_by_source",
        "total_effective_healing_by_recipient",
        "health_after_combat_resolution_by_recipient",
    )
    payload: dict[str, object] = {}
    for field_name in bool_vectors:
        payload[field_name] = _array_payload(
            getattr(source, field_name),
            name=f"combat_transition_facts.{field_name}",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_BOOL_DTYPE,
        )
    for field_name in float_vectors:
        payload[field_name] = _array_payload(
            getattr(source, field_name),
            name=f"combat_transition_facts.{field_name}",
            shape=(MAX_AGENT_SLOTS,),
            dtype=_FLOAT32_DTYPE,
            finite=True,
        )
    for field_name, width in (
        ("slow_is_applied_by_source_and_channel", NUM_SLOW_CHANNELS),
        ("stun_is_applied_by_source_and_channel", NUM_STUN_CHANNELS),
    ):
        payload[field_name] = _array_payload(
            getattr(source, field_name),
            name=f"combat_transition_facts.{field_name}",
            shape=(MAX_AGENT_SLOTS, width),
            dtype=_BOOL_DTYPE,
        )

    has_recipient = _require_host_array(
        source.combat_effect_has_recipient_by_source,
        name="combat_transition_facts.combat_effect_has_recipient_by_source",
        shape=(MAX_AGENT_SLOTS,),
        dtype=_BOOL_DTYPE,
    )
    recipient_slots = _require_host_array(
        source.combat_effect_recipient_global_slot_by_source,
        name=("combat_transition_facts.combat_effect_recipient_global_slot_by_source"),
        shape=(MAX_AGENT_SLOTS,),
        dtype=_INT32_DTYPE,
    )
    normalized_recipients: list[int | None] = []
    for source_slot, (has_value, recipient_value) in enumerate(
        zip(has_recipient, recipient_slots, strict=True)
    ):
        has_recipient_value = bool(has_value)
        recipient = int(recipient_value)
        if has_recipient_value:
            if not 0 <= recipient < MAX_AGENT_SLOTS:
                raise ValueError(
                    f"combat recipient for source {source_slot} must be in [0, 10)"
                )
            normalized_recipients.append(recipient)
        else:
            if recipient != -1:
                raise ValueError(
                    f"recipient-less source {source_slot} must use core sentinel -1"
                )
            normalized_recipients.append(None)
    payload["combat_effect_recipient_global_slot_by_source"] = tuple(
        normalized_recipients
    )
    return CombatTransitionFactsV1.model_validate(payload)


def _normalize_simple_fact_model(
    source: object,
    *,
    source_type: type[object],
    source_name: str,
    model_type: type[
        DeathTransitionFactsV1
        | SpawnShieldTransitionFactsV1
        | RespawnTransitionFactsV1
        | RegenerationTransitionFactsV1
        | PhysicalTransitionFactsV1
        | AuraTransitionFactsV1
        | StatusLifecycleTransitionFactsV1
        | TeamDeathmatchTransitionFactsV1
    ],
    specs: tuple[tuple[str, tuple[int, ...], np.dtype[np.generic], bool], ...],
) -> (
    DeathTransitionFactsV1
    | SpawnShieldTransitionFactsV1
    | RespawnTransitionFactsV1
    | RegenerationTransitionFactsV1
    | PhysicalTransitionFactsV1
    | AuraTransitionFactsV1
    | StatusLifecycleTransitionFactsV1
    | TeamDeathmatchTransitionFactsV1
):
    """Build one fact subtree from explicit host field/shape/dtype specifications.

    Require the exact source record type and validate every named NumPy leaf before
    constructing model_type. The specs boolean controls finite-value checking.
    This shared adapter copies facts without deriving domain rules.
    """
    _require_exact_type(source, source_type, name=source_name)
    payload = {
        field_name: _array_payload(
            getattr(source, field_name),
            name=f"{source_name}.{field_name}",
            shape=shape,
            dtype=dtype,
            finite=finite,
        )
        for field_name, shape, dtype, finite in specs
    }
    return model_type.model_validate(payload)


def normalize_transition_facts_v1(source: TransitionFacts) -> TransitionFactsV1:
    """Convert an already-host Core fact tree into its strict V1 wire record.

    Parameters
    ----------
    source : TransitionFacts
        Exact unbatched TransitionFacts returned by a bundled jax.device_get.
        Every leaf must be an exact NumPy array with Core's fixed shape and
        bool, int32, or float32 dtype.

    Returns
    -------
    TransitionFactsV1
        Strict V1 fact tree with immutable Python values. Missing recipient routes
        become None; valid numeric values and all independent cause flags are retained.

    Raises
    ------
    TypeError
        A record type/dtype is wrong, or a leaf is still a JAX array.
    ValueError
        A shape, finite-value check, accepted action category, recipient
        route, or initialization/transition tick is invalid.

    Notes
    -----
    Performs no transfer. This prevents a tree walk from causing repeated
    implicit device-to-host copies. Initialization facts are allowed with
    has_transition false and start tick -1; actual transitions require a
    nonnegative start tick. Transition capture separately rejects initialization.
    """
    _require_exact_type(source, TransitionFacts, name="transition_facts")
    has_transition = cast(
        bool,
        _array_payload(
            source.has_transition,
            name="transition_facts.has_transition",
            shape=(),
            dtype=_BOOL_DTYPE,
        ),
    )
    transition_start_step_count = cast(
        int,
        _array_payload(
            source.transition_start_step_count,
            name="transition_facts.transition_start_step_count",
            shape=(),
            dtype=_INT32_DTYPE,
        ),
    )
    if has_transition and transition_start_step_count < 0:
        raise ValueError("real transition facts require a nonnegative start step")
    if not has_transition and transition_start_step_count != -1:
        raise ValueError(
            "initialization facts require the canonical start-step sentinel -1"
        )

    death = cast(
        DeathTransitionFactsV1,
        _normalize_simple_fact_model(
            source.death_facts,
            source_type=DeathTransitionFacts,
            source_name="death_facts",
            model_type=DeathTransitionFactsV1,
            specs=(
                ("is_newly_dead_by_recipient", (MAX_AGENT_SLOTS,), _BOOL_DTYPE, False),
                (
                    "contributed_to_new_death_by_source",
                    (MAX_AGENT_SLOTS,),
                    _BOOL_DTYPE,
                    False,
                ),
                (
                    "attributed_death_damage_by_source",
                    (MAX_AGENT_SLOTS,),
                    _FLOAT32_DTYPE,
                    True,
                ),
            ),
        ),
    )
    spawn_shield = cast(
        SpawnShieldTransitionFactsV1,
        _normalize_simple_fact_model(
            source.spawn_shield_facts,
            source_type=SpawnShieldTransitionFacts,
            source_name="spawn_shield_facts",
            model_type=SpawnShieldTransitionFactsV1,
            specs=tuple(
                (field_name, (MAX_AGENT_SLOTS,), _BOOL_DTYPE, False)
                for field_name in SpawnShieldTransitionFacts._fields
            ),
        ),
    )
    respawn = cast(
        RespawnTransitionFactsV1,
        _normalize_simple_fact_model(
            source.respawn_facts,
            source_type=RespawnTransitionFacts,
            source_name="respawn_facts",
            model_type=RespawnTransitionFactsV1,
            specs=(
                (
                    "respawn_wave_occurred_this_transition_by_team",
                    (NUM_TEAMS,),
                    _BOOL_DTYPE,
                    False,
                ),
                (
                    "was_respawned_this_transition_by_agent",
                    (MAX_AGENT_SLOTS,),
                    _BOOL_DTYPE,
                    False,
                ),
            ),
        ),
    )
    regeneration = cast(
        RegenerationTransitionFactsV1,
        _normalize_simple_fact_model(
            source.regeneration_facts,
            source_type=RegenerationTransitionFacts,
            source_name="regeneration_facts",
            model_type=RegenerationTransitionFactsV1,
            specs=(
                (
                    "combat_countdown_was_reset_by_agent",
                    (MAX_AGENT_SLOTS,),
                    _BOOL_DTYPE,
                    False,
                ),
                (
                    "actual_health_regenerated_this_step_by_agent",
                    (MAX_AGENT_SLOTS,),
                    _FLOAT32_DTYPE,
                    True,
                ),
            ),
        ),
    )
    physical = cast(
        PhysicalTransitionFactsV1,
        _normalize_simple_fact_model(
            source.physical_facts,
            source_type=PhysicalTransitionFacts,
            source_name="physical_facts",
            model_type=PhysicalTransitionFactsV1,
            specs=tuple(
                (
                    field_name,
                    (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
                    _FLOAT32_DTYPE,
                    True,
                )
                for field_name in PhysicalTransitionFacts._fields
            ),
        ),
    )
    aura = cast(
        AuraTransitionFactsV1,
        _normalize_simple_fact_model(
            source.aura_facts,
            source_type=AuraTransitionFacts,
            source_name="aura_facts",
            model_type=AuraTransitionFactsV1,
            specs=tuple(
                (
                    field_name,
                    (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
                    _BOOL_DTYPE,
                    False,
                )
                for field_name in AuraTransitionFacts._fields
            ),
        ),
    )
    status_lifecycle = cast(
        StatusLifecycleTransitionFactsV1,
        _normalize_simple_fact_model(
            source.status_lifecycle_facts,
            source_type=StatusLifecycleTransitionFacts,
            source_name="status_lifecycle_facts",
            model_type=StatusLifecycleTransitionFactsV1,
            specs=tuple(
                (
                    field_name,
                    (MAX_AGENT_SLOTS, _NUM_STATUS_CHANNELS),
                    _BOOL_DTYPE,
                    False,
                )
                for field_name in StatusLifecycleTransitionFacts._fields
            ),
        ),
    )
    team_deathmatch = cast(
        TeamDeathmatchTransitionFactsV1,
        _normalize_simple_fact_model(
            source.team_deathmatch_facts,
            source_type=TeamDeathmatchTransitionFacts,
            source_name="team_deathmatch_facts",
            model_type=TeamDeathmatchTransitionFactsV1,
            specs=(("outcome", (), _INT32_DTYPE, False),),
        ),
    )

    return TransitionFactsV1.model_validate(
        {
            "has_transition": has_transition,
            "transition_start_step_count": transition_start_step_count,
            "action_acceptance_facts": _normalize_action_acceptance_facts_v1(
                source.action_acceptance_facts
            ),
            "combat_transition_facts": _normalize_combat_transition_facts_v1(
                source.combat_transition_facts
            ),
            "death_facts": death,
            "spawn_shield_facts": spawn_shield,
            "respawn_facts": respawn,
            "regeneration_facts": regeneration,
            "physical_facts": physical,
            "aura_facts": aura,
            "status_lifecycle_facts": status_lifecycle,
            "team_deathmatch_facts": team_deathmatch,
        }
    )


def _build_evaluation_frame_from_host(
    context: EvaluationEpisodeContext,
    *,
    frame_index: int,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (object | None),
) -> EvaluationFrame:
    """Build a canonical indexed frame from a bundle already copied to NumPy.

    Require a nonnegative exact Python frame_index. Choose the frame version that
    pairs with the context (evaluation_frame_type_for_context: V3 for context V4,
    V2 for V3, otherwise V1); normalize snapshot, observation, masks, and
    optional (10, 10) SharedObs availability, then check information sharing.
    No second transfer
    occurs. Full cross-record validation is owned by replay admission.
    """
    evaluation_context_type(context)
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative exact integer")
    simulator_step_count, snapshot = _normalize_snapshot_v1(state)
    availability_payload: object | None = None
    if shared_obs_information_availability_by_recipient_and_sensor_source is not None:
        availability_payload = _array_payload(
            shared_obs_information_availability_by_recipient_and_sensor_source,
            name=("shared_obs_information_availability_by_recipient_and_sensor_source"),
            shape=(MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
            dtype=_BOOL_DTYPE,
        )

    episode_id = context.identity.episode_id
    frame_model = evaluation_frame_type_for_context(context)
    frame = frame_model.model_validate(
        {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "frame_id": f"{episode_id}:frame:{frame_index}",
            "simulator_step_count": simulator_step_count,
            "snapshot": snapshot,
            "base_observation": _normalize_base_observation(observation, context),
            "action_mask": _normalize_action_mask_v1(action_mask),
            "shared_obs_information_availability_by_recipient_and_sensor_source": (
                availability_payload
            ),
        }
    )
    _validate_frame_information_regime(context, frame)
    return frame


def _capture_initial_evaluation_frame(
    context: EvaluationEpisodeContext,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> EvaluationFrame:
    """Copy one complete initial bundle, then build artifact frame zero.

    Use one jax.device_get call for state, observation, masks, and optional
    availability. Context V1 also receives full initial-frame validation here;
    later contexts are fully checked by the enclosing replay admission path.
    """
    evaluation_context_type(context)
    host_state, host_observation, host_action_mask, host_availability = cast(
        tuple[EnvState, Observation, ActionMask, object | None],
        jax.device_get(
            (
                state,
                observation,
                action_mask,
                shared_obs_information_availability_by_recipient_and_sensor_source,
            )
        ),
    )
    frame = _build_evaluation_frame_from_host(
        context,
        frame_index=0,
        state=host_state,
        observation=host_observation,
        action_mask=host_action_mask,
        shared_obs_information_availability_by_recipient_and_sensor_source=(
            host_availability
        ),
    )
    if type(context) is EvaluationEpisodeContextV1:
        validate_initial_evaluation_frame_v1(context, frame)
    return frame


def _normalize_reward_v1(source: Reward) -> object:
    """Freeze ten finite float32 canonical rewards from an exact host Reward.

    Return a Python tuple in global-slot order; task consistency is checked after
    joining these values to the transition's task facts.
    """
    _require_exact_type(source, Reward, name="canonical_reward")
    return _array_payload(
        source.rewards,
        name="canonical_reward.rewards",
        shape=(MAX_AGENT_SLOTS,),
        dtype=_FLOAT32_DTYPE,
        finite=True,
    )


def _normalize_done_flags_v1(source: DoneFlags) -> tuple[bool, bool]:
    """Return terminated and truncated from exact host scalar bool arrays.

    Require an exact DoneFlags record. Keep the two flags independent because both
    may be true on the same transition.
    """
    _require_exact_type(source, DoneFlags, name="done_flags")
    terminated = cast(
        bool,
        _array_payload(
            source.terminated,
            name="done_flags.terminated",
            shape=(),
            dtype=_BOOL_DTYPE,
        ),
    )
    truncated = cast(
        bool,
        _array_payload(
            source.truncated,
            name="done_flags.truncated",
            shape=(),
            dtype=_BOOL_DTYPE,
        ),
    )
    return terminated, truncated


def _capture_evaluation_transition_unit(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    successor_state: EnvState,
    successor_observation: Observation,
    successor_action_mask: ActionMask,
    transition_facts: TransitionFacts,
    canonical_reward: Reward,
    done_flags: DoneFlags,
    *,
    successor_shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> tuple[EvaluationTransitionV1, EvaluationFrame]:
    """Copy a successor/facts/reward bundle and build its joined transition record.

    One jax.device_get call transfers all supplied outputs. Normalize the successor,
    require actual transition facts, check task rewards/completion, and decode
    canonical events. Context V1 additionally gets full transition-unit validation
    here; replay admission validates later context versions. Return transition then
    successor frame without modifying the previous frame.
    """
    evaluation_context_type(context)
    evaluation_frame_type(start_frame)
    (
        host_successor_state,
        host_successor_observation,
        host_successor_action_mask,
        host_transition_facts,
        host_canonical_reward,
        host_done_flags,
        host_successor_availability,
    ) = cast(
        tuple[
            EnvState,
            Observation,
            ActionMask,
            TransitionFacts,
            Reward,
            DoneFlags,
            object | None,
        ],
        jax.device_get(
            (
                successor_state,
                successor_observation,
                successor_action_mask,
                transition_facts,
                canonical_reward,
                done_flags,
                successor_shared_obs_information_availability_by_recipient_and_sensor_source,
            )
        ),
    )

    successor_frame = _build_evaluation_frame_from_host(
        context,
        frame_index=start_frame.frame_index + 1,
        state=host_successor_state,
        observation=host_successor_observation,
        action_mask=host_successor_action_mask,
        shared_obs_information_availability_by_recipient_and_sensor_source=(
            host_successor_availability
        ),
    )
    facts = normalize_transition_facts_v1(host_transition_facts)
    if not facts.has_transition:
        raise ValueError("evaluation transition capture rejects initialization facts")
    canonical_reward_by_agent = cast(
        tuple[float, ...],
        _normalize_reward_v1(host_canonical_reward),
    )
    terminated, truncated = _normalize_done_flags_v1(host_done_flags)
    canonical_reward_by_team, owning_task_end_reason = (
        _derive_and_validate_team_deathmatch_authority_v1(
            context,
            start_frame,
            facts,
            successor_frame,
            canonical_reward_by_agent,
            terminated,
            truncated,
        )
    )
    transition_index = start_frame.frame_index
    episode_id = context.identity.episode_id
    transition_id = f"{episode_id}:transition:{transition_index}"
    events = decode_evaluation_events_v1(
        context,
        start_frame,
        facts,
        successor_frame,
    )
    transition = EvaluationTransitionV1.model_validate(
        {
            "episode_id": episode_id,
            "transition_index": transition_index,
            "transition_id": transition_id,
            "start_frame_id": start_frame.frame_id,
            "successor_frame_id": successor_frame.frame_id,
            "facts": facts,
            "events": events,
            "canonical_reward_by_agent": canonical_reward_by_agent,
            "canonical_reward_by_team": canonical_reward_by_team,
            "terminated": terminated,
            "truncated": truncated,
            "owning_task_end_reason": owning_task_end_reason,
        }
    )
    if type(context) is EvaluationEpisodeContextV1:
        validate_evaluation_transition_unit_v1(
            context,
            start_frame,
            transition,
            successor_frame,
        )
    return transition, successor_frame


def reconstruct_env_state_v1(frame: EvaluationFrame, *, host: bool = False) -> EnvState:
    """Restore the simulator state recorded in one captured frame.

    Parameters
    ----------
    frame : EvaluationFrame
        Supported, validated frame containing the full analysis snapshot.
    host : bool
        If false, return JAX leaves on the default device. If true, return
        NumPy leaves so callers can batch them before a later transfer. Defaults to
        false.

    Returns
    -------
    EnvState
        Unbatched EnvState with the recorded simulator tick, ten global slots,
        two team rows, float32 position/health, int32 counters/actions, and bool flags.

    Raises
    ------
    TypeError
        A recorded integer field contains a non-integer value.
    ValueError
        Integers exceed int32 or floating values cannot be represented
        exactly as finite float32.

    Notes
    -----
    This host function restores values without resetting, stepping, or inferring
    state from observations. Core NamedTuples preserve the PyTree structure;
    host=True leaves are NumPy arrays despite the runtime type annotations.
    """
    from marl_battlegrounds.evaluation.catalog import (
        _wire_float32_array,  # pyright: ignore[reportPrivateUsage]
        _wire_int32_array,  # pyright: ignore[reportPrivateUsage]
    )

    values = frame.snapshot.model_dump(exclude={"schema_id", "schema_version"})
    values["step_count"] = frame.simulator_step_count
    arrays: dict[str, jax.Array] = {}
    for name in EnvState._fields:
        value = values[name]
        if name in ("alive_mask", "has_previous_timestep_joint_action"):
            array = np.asarray(value, dtype=np.bool_)
            arrays[name] = cast(jax.Array, array) if host else jnp.asarray(array)
        else:
            convert = (
                _wire_float32_array
                if name in ("agent_positions", "current_health")
                else _wire_int32_array
            )
            arrays[name] = convert(
                value, field_name=f"frame.snapshot.{name}", host=host
            )
    return EnvState(**arrays)


def _reconstruct_transition_facts(
    source: TransitionFactsV1,
    *,
    host: bool = False,
) -> TransitionFacts:
    """Restore Core fact arrays from one already validated wire record.

    Parameters
    ----------
    source : TransitionFactsV1
        Exact TransitionFactsV1 with fixed global-slot and channel axes.
    host : bool
        If false, construct JAX arrays on the default device. If true, return
        NumPy arrays for later batched transfer. Defaults to false.

    Returns
    -------
    TransitionFacts
        Unbatched Core TransitionFacts with bool, int32, and float32 leaves.
        Absent recipient routes are restored from None to Core's -1 sentinel.

    Raises
    ------
    TypeError
        source is not the exact declared V1 record type.

    Notes
    -----
    Exported as reconstruct_transition_facts_v1. It does not revalidate the whole
    wire tree, check lossless numeric narrowing, or simulate effects. Admission
    must validate records before reconstruction. host=True avoids per-field
    device dispatch while preserving the Core NamedTuple structure.
    """

    def array(value: object, *, dtype: DTypeLike) -> jax.Array:
        """Construct a NumPy or JAX leaf according to the enclosing host flag.

        Use the explicitly supplied dtype. Source values have already passed wire
        admission; this conversion does not repeat semantic validation.
        """
        if host:
            return cast(jax.Array, np.asarray(value, dtype=dtype))
        return jnp.asarray(value, dtype=np.dtype(dtype))

    _require_exact_type(source, TransitionFactsV1, name="transition_facts_v1")

    def action(model: JointActionV1) -> Action:
        """Restore the three recorded action heads as length-ten int32 arrays.

        Use the enclosing host/device conversion choice and preserve submitted values.
        """
        return Action(
            move=array(model.move, dtype=jnp.int32),
            select_target=array(model.select_target, dtype=jnp.int32),
            use_ultimate=array(model.use_ultimate, dtype=jnp.int32),
        )

    acceptance = source.action_acceptance_facts
    action_acceptance_facts = ActionAcceptanceFacts(
        submitted_joint_action=action(acceptance.submitted_joint_action),
        accepted_joint_action=action(acceptance.accepted_joint_action),
        submitted_action_tuple_is_out_of_domain_by_actor=array(
            acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
            dtype=jnp.bool_,
        ),
        in_domain_move_action_is_rejected_by_actor=array(
            acceptance.in_domain_move_action_is_rejected_by_actor,
            dtype=jnp.bool_,
        ),
        in_domain_combat_action_pair_is_rejected_by_actor=array(
            acceptance.in_domain_combat_action_pair_is_rejected_by_actor,
            dtype=jnp.bool_,
        ),
    )

    combat = source.combat_transition_facts
    recipient_slots = tuple(
        -1 if recipient is None else recipient
        for recipient in combat.combat_effect_recipient_global_slot_by_source
    )
    combat_transition_facts = CombatTransitionFacts(
        basic_effect_is_activated_by_source=array(
            combat.basic_effect_is_activated_by_source, dtype=jnp.bool_
        ),
        ultimate_effect_is_activated_by_source=array(
            combat.ultimate_effect_is_activated_by_source, dtype=jnp.bool_
        ),
        combat_effect_has_recipient_by_source=array(
            combat.combat_effect_has_recipient_by_source, dtype=jnp.bool_
        ),
        combat_effect_recipient_global_slot_by_source=array(
            recipient_slots, dtype=jnp.int32
        ),
        raw_damage_output_by_source=array(
            combat.raw_damage_output_by_source, dtype=jnp.float32
        ),
        source_modified_damage_output_by_source=array(
            combat.source_modified_damage_output_by_source, dtype=jnp.float32
        ),
        recipient_damage_modifier_by_source=array(
            combat.recipient_damage_modifier_by_source, dtype=jnp.float32
        ),
        total_effective_damage_by_recipient=array(
            combat.total_effective_damage_by_recipient, dtype=jnp.float32
        ),
        raw_healing_output_by_source=array(
            combat.raw_healing_output_by_source, dtype=jnp.float32
        ),
        source_modified_healing_output_by_source=array(
            combat.source_modified_healing_output_by_source, dtype=jnp.float32
        ),
        recipient_healing_modifier_by_source=array(
            combat.recipient_healing_modifier_by_source, dtype=jnp.float32
        ),
        total_effective_healing_by_recipient=array(
            combat.total_effective_healing_by_recipient, dtype=jnp.float32
        ),
        health_after_combat_resolution_by_recipient=array(
            combat.health_after_combat_resolution_by_recipient, dtype=jnp.float32
        ),
        slow_is_applied_by_source_and_channel=array(
            combat.slow_is_applied_by_source_and_channel, dtype=jnp.bool_
        ),
        stun_is_applied_by_source_and_channel=array(
            combat.stun_is_applied_by_source_and_channel, dtype=jnp.bool_
        ),
        rogue_poison_anti_heal_is_applied_by_source=array(
            combat.rogue_poison_anti_heal_is_applied_by_source, dtype=jnp.bool_
        ),
        mage_burst_damage_amplification_is_applied_by_source=array(
            combat.mage_burst_damage_amplification_is_applied_by_source,
            dtype=jnp.bool_,
        ),
        priest_blessing_of_freedom_is_applied_by_source=array(
            combat.priest_blessing_of_freedom_is_applied_by_source,
            dtype=jnp.bool_,
        ),
    )
    death = source.death_facts
    shield = source.spawn_shield_facts
    respawn = source.respawn_facts
    regeneration = source.regeneration_facts
    physical = source.physical_facts
    aura = source.aura_facts
    lifecycle = source.status_lifecycle_facts
    team_deathmatch = source.team_deathmatch_facts
    return TransitionFacts(
        has_transition=array(source.has_transition, dtype=jnp.bool_),
        transition_start_step_count=array(
            source.transition_start_step_count, dtype=jnp.int32
        ),
        action_acceptance_facts=action_acceptance_facts,
        combat_transition_facts=combat_transition_facts,
        death_facts=DeathTransitionFacts(
            array(death.is_newly_dead_by_recipient, dtype=jnp.bool_),
            array(death.contributed_to_new_death_by_source, dtype=jnp.bool_),
            array(death.attributed_death_damage_by_source, dtype=jnp.float32),
        ),
        spawn_shield_facts=SpawnShieldTransitionFacts(
            array(shield.was_active_at_transition_start_by_agent, dtype=jnp.bool_),
            array(shield.expired_at_transition_end_by_agent, dtype=jnp.bool_),
        ),
        respawn_facts=RespawnTransitionFacts(
            array(
                respawn.respawn_wave_occurred_this_transition_by_team,
                dtype=jnp.bool_,
            ),
            array(
                respawn.was_respawned_this_transition_by_agent,
                dtype=jnp.bool_,
            ),
        ),
        regeneration_facts=RegenerationTransitionFacts(
            array(regeneration.combat_countdown_was_reset_by_agent, dtype=jnp.bool_),
            array(
                regeneration.actual_health_regenerated_this_step_by_agent,
                dtype=jnp.float32,
            ),
        ),
        physical_facts=PhysicalTransitionFacts(
            array(physical.charge_phase_displacement_by_agent, dtype=jnp.float32),
            array(
                physical.ordinary_movement_phase_displacement_by_agent,
                dtype=jnp.float32,
            ),
        ),
        aura_facts=AuraTransitionFacts(
            array(
                aura.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
                dtype=jnp.bool_,
            ),
            array(
                aura.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
                dtype=jnp.bool_,
            ),
        ),
        status_lifecycle_facts=StatusLifecycleTransitionFacts(
            array(
                lifecycle.aged_to_zero_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            array(
                lifecycle.refreshed_or_extended_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            array(
                lifecycle.broken_by_damage_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            array(
                lifecycle.cleared_by_new_death_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
        ),
        team_deathmatch_facts=TeamDeathmatchTransitionFacts(
            outcome=array(team_deathmatch.outcome, dtype=jnp.int32),
        ),
    )


reconstruct_transition_facts_v1 = _reconstruct_transition_facts


__all__ = [
    "capture_evaluation_transition_unit_v1",
    "capture_evaluation_transition_unit_v2",
    "capture_evaluation_transition_unit_v3",
    "capture_initial_evaluation_frame_v1",
    "capture_initial_evaluation_frame_v2",
    "capture_initial_evaluation_frame_v3",
    "normalize_transition_facts_v1",
    "reconstruct_env_state_v1",
    "reconstruct_transition_facts_v1",
]


def capture_initial_evaluation_frame_v1(
    context: EvaluationEpisodeContext,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> EvaluationFrameV1:
    """Capture historical artifact frame zero from simulator outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V1 or V2 for historical frame V1.
    state : EnvState
        Unbatched EnvState at the captured decision epoch. Its simulator tick
        may be nonzero.
    observation : Observation
        Matching unbatched Core Observation for all ten global slots.
    action_mask : ActionMask
        Matching unbatched Core ActionMask with boolean leaves.
    shared_obs_information_availability_by_recipient_and_sensor_source : object | None
        Optional
        bool array shaped (10, 10), recipient then sensor source. Required for
        SharedObs and omitted for NoSharedObs. Defaults to None.

    Returns
    -------
    EvaluationFrameV1
        Immutable frame V1 with artifact index zero and canonical episode/frame ID.

    Raises
    ------
    TypeError
        Core records or leaf dtypes are wrong after host transfer.
    ValueError
        Context version, shapes, feature values, class/roster mapping,
        or information availability is inconsistent.

    Notes
    -----
    Runs outside JIT and synchronizes a complete bundle through one device_get
    call. It does not step or reset the environment. Legacy observations retain
    historical Team ID features; current observations must use V3 capture.
    """
    if type(context) in (EvaluationEpisodeContextV3, EvaluationEpisodeContextV4):
        raise ValueError("capture V1 requires matching episode context")
    return cast(
        EvaluationFrameV1,
        _capture_initial_evaluation_frame(
            context,
            state,
            observation,
            action_mask,
            shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )


def capture_initial_evaluation_frame_v2(
    context: EvaluationEpisodeContext,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> EvaluationFrameV2:
    """Capture frame zero of a pre-Red-Zone (context V3) recording from outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V3 for frame V2 (before the Red Zone rule; context V4
        uses capture_initial_evaluation_frame_v3).
    state : EnvState
        Unbatched EnvState at the captured decision epoch. Its simulator tick
        may be nonzero.
    observation : Observation
        Matching unbatched Core Observation for all ten global slots.
    action_mask : ActionMask
        Matching unbatched Core ActionMask with boolean leaves.
    shared_obs_information_availability_by_recipient_and_sensor_source : object | None
        Optional
        bool array shaped (10, 10), recipient then sensor source. Required for
        SharedObs and omitted for NoSharedObs. Defaults to None.

    Returns
    -------
    EvaluationFrameV2
        Immutable frame V2 with artifact index zero and canonical episode/frame ID.

    Raises
    ------
    TypeError
        Core records or leaf dtypes are wrong after host transfer.
    ValueError
        Context version, shapes, feature values, class/roster mapping,
        or information availability is inconsistent.

    Notes
    -----
    Runs outside JIT and synchronizes a complete bundle through one device_get
    call. It does not step or reset the environment. Full initial-frame/context
    revalidation is performed by replay admission.
    """
    if type(context) is not EvaluationEpisodeContextV3:
        raise ValueError("capture V2 requires matching episode context")
    return cast(
        EvaluationFrameV2,
        _capture_initial_evaluation_frame(
            context,
            state,
            observation,
            action_mask,
            shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )


def capture_initial_evaluation_frame_v3(
    context: EvaluationEpisodeContext,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> EvaluationFrameV3:
    """Capture current 20-column artifact frame zero from simulator outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V4 for current frame V3.
    state : EnvState
        Unbatched EnvState at the captured decision epoch. Its simulator tick
        may be nonzero.
    observation : Observation
        Matching unbatched Core Observation for all ten global slots.
    action_mask : ActionMask
        Matching unbatched Core ActionMask with boolean leaves.
    shared_obs_information_availability_by_recipient_and_sensor_source : object | None
        Optional
        bool array shaped (10, 10), recipient then sensor source. Required for
        SharedObs and omitted for NoSharedObs. Defaults to None.

    Returns
    -------
    EvaluationFrameV3
        Immutable frame V3 with artifact index zero and canonical episode/frame ID.
        Context column 19 on every active row is the configured Red Zone depth.

    Raises
    ------
    TypeError
        Core records or leaf dtypes are wrong after host transfer.
    ValueError
        Context version, shapes, feature values, class/roster mapping,
        or information availability is inconsistent.

    Notes
    -----
    Runs outside JIT and synchronizes a complete bundle through one device_get
    call. It does not step or reset the environment. Full initial-frame/context
    revalidation is performed by replay admission.
    """
    if type(context) is not EvaluationEpisodeContextV4:
        raise ValueError("capture V3 requires matching episode context")
    return cast(
        EvaluationFrameV3,
        _capture_initial_evaluation_frame(
            context,
            state,
            observation,
            action_mask,
            shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )


def capture_evaluation_transition_unit_v1(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrameV1,
    successor_state: EnvState,
    successor_observation: Observation,
    successor_action_mask: ActionMask,
    transition_facts: TransitionFacts,
    canonical_reward: Reward,
    done_flags: DoneFlags,
    *,
    successor_shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> tuple[EvaluationTransitionV1, EvaluationFrameV1]:
    """Capture a transition and historical successor frame from simulator outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V1 or V2 for frame V1.
    start_frame : EvaluationFrameV1
        Matching frame V1 already captured for the transition's decision epoch.
    successor_state : EnvState
        Unbatched EnvState after exactly one simulator tick.
    successor_observation : Observation
        Matching successor Core Observation for ten global slots.
    successor_action_mask : ActionMask
        Matching successor Core ActionMask.
    transition_facts : TransitionFacts
        Actual unbatched Core TransitionFacts from this step;
        initialization facts are rejected.
    canonical_reward : Reward
        Unbatched Reward with ten finite float32 task rewards.
    done_flags : DoneFlags
        Scalar bool terminated and truncated flags from this same step.
    successor_shared_obs_information_availability_by_recipient_and_sensor_source : Array
        Optional successor bool array shaped (10, 10), recipient then sensor
        source. Required for SharedObs, omitted for NoSharedObs; defaults to None.

    Returns
    -------
    tuple[EvaluationTransitionV1, EvaluationFrameV1]
        Pair of EvaluationTransitionV1 and EvaluationFrameV1. The transition uses
        start_frame's artifact index and the successor uses the next index.

    Raises
    ------
    TypeError
        Core record types or leaf dtypes are wrong after host transfer.
    ValueError
        Versions, adjacency, facts, rewards, completion, event joins,
        or successor information availability are inconsistent.

    Notes
    -----
    One host device_get call handles the complete supplied output bundle.
    start_frame is reused without transfer or mutation. This does not step the
    simulator. Full strict-tree admission is also performed when building or
    reading the complete replay; capture alone is not an artifact integrity audit.
    """
    if (
        type(context) in (EvaluationEpisodeContextV3, EvaluationEpisodeContextV4)
        or type(start_frame) is not EvaluationFrameV1
    ):
        raise ValueError("capture V1 requires matching episode context and frame")
    return cast(
        tuple[EvaluationTransitionV1, EvaluationFrameV1],
        _capture_evaluation_transition_unit(
            context,
            start_frame,
            successor_state,
            successor_observation,
            successor_action_mask,
            transition_facts,
            canonical_reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=successor_shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )


def capture_evaluation_transition_unit_v2(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrameV2,
    successor_state: EnvState,
    successor_observation: Observation,
    successor_action_mask: ActionMask,
    transition_facts: TransitionFacts,
    canonical_reward: Reward,
    done_flags: DoneFlags,
    *,
    successor_shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> tuple[EvaluationTransitionV1, EvaluationFrameV2]:
    """Capture a transition and its V2 successor frame (context V3) from outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V3 for frame V2.
    start_frame : EvaluationFrameV2
        Matching frame V2 already captured for the transition's decision epoch.
    successor_state : EnvState
        Unbatched EnvState after exactly one simulator tick.
    successor_observation : Observation
        Matching successor Core Observation for ten global slots.
    successor_action_mask : ActionMask
        Matching successor Core ActionMask.
    transition_facts : TransitionFacts
        Actual unbatched Core TransitionFacts from this step;
        initialization facts are rejected.
    canonical_reward : Reward
        Unbatched Reward with ten finite float32 task rewards.
    done_flags : DoneFlags
        Scalar bool terminated and truncated flags from this same step.
    successor_shared_obs_information_availability_by_recipient_and_sensor_source : Array
        Optional successor bool array shaped (10, 10), recipient then sensor
        source. Required for SharedObs, omitted for NoSharedObs; defaults to None.

    Returns
    -------
    tuple[EvaluationTransitionV1, EvaluationFrameV2]
        Pair of EvaluationTransitionV1 and EvaluationFrameV2. The transition uses
        start_frame's artifact index and the successor uses the next index.

    Raises
    ------
    TypeError
        Core record types or leaf dtypes are wrong after host transfer.
    ValueError
        Versions, adjacency, facts, rewards, completion, event joins,
        or successor information availability are inconsistent.

    Notes
    -----
    One host device_get call handles the complete supplied output bundle.
    start_frame is reused without transfer or mutation. This does not step the
    simulator. Full strict-tree admission is also performed when building or
    reading the complete replay; capture alone is not an artifact integrity audit.
    """
    if (
        type(context) is not EvaluationEpisodeContextV3
        or type(start_frame) is not EvaluationFrameV2
    ):
        raise ValueError("capture V2 requires matching episode context and frame")
    return cast(
        tuple[EvaluationTransitionV1, EvaluationFrameV2],
        _capture_evaluation_transition_unit(
            context,
            start_frame,
            successor_state,
            successor_observation,
            successor_action_mask,
            transition_facts,
            canonical_reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=successor_shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )


def capture_evaluation_transition_unit_v3(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrameV3,
    successor_state: EnvState,
    successor_observation: Observation,
    successor_action_mask: ActionMask,
    transition_facts: TransitionFacts,
    canonical_reward: Reward,
    done_flags: DoneFlags,
    *,
    successor_shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> tuple[EvaluationTransitionV1, EvaluationFrameV3]:
    """Capture a transition and current 20-column successor frame from Core outputs.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact context V4 for frame V3.
    start_frame : EvaluationFrameV3
        Matching frame V3 already captured for the transition's decision epoch.
    successor_state : EnvState
        Unbatched EnvState after exactly one simulator tick.
    successor_observation : Observation
        Matching successor Core Observation for ten global slots.
    successor_action_mask : ActionMask
        Matching successor Core ActionMask.
    transition_facts : TransitionFacts
        Actual unbatched Core TransitionFacts from this step;
        initialization facts are rejected.
    canonical_reward : Reward
        Unbatched Reward with ten finite float32 task rewards.
    done_flags : DoneFlags
        Scalar bool terminated and truncated flags from this same step.
    successor_shared_obs_information_availability_by_recipient_and_sensor_source : Array
        Optional successor bool array shaped (10, 10), recipient then sensor
        source. Required for SharedObs, omitted for NoSharedObs; defaults to None.

    Returns
    -------
    tuple[EvaluationTransitionV1, EvaluationFrameV3]
        Pair of EvaluationTransitionV1 and EvaluationFrameV3. The transition uses
        start_frame's artifact index and the successor uses the next index.

    Raises
    ------
    TypeError
        Core record types or leaf dtypes are wrong after host transfer.
    ValueError
        Versions, adjacency, facts, rewards, completion, event joins,
        or successor information availability are inconsistent.

    Notes
    -----
    One host device_get call handles the complete supplied output bundle.
    start_frame is reused without transfer or mutation. This does not step the
    simulator. Full strict-tree admission is also performed when building or
    reading the complete replay; capture alone is not an artifact integrity audit.
    """
    if (
        type(context) is not EvaluationEpisodeContextV4
        or type(start_frame) is not EvaluationFrameV3
    ):
        raise ValueError("capture V3 requires matching episode context and frame")
    return cast(
        tuple[EvaluationTransitionV1, EvaluationFrameV3],
        _capture_evaluation_transition_unit(
            context,
            start_frame,
            successor_state,
            successor_observation,
            successor_action_mask,
            transition_facts,
            canonical_reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=successor_shared_obs_information_availability_by_recipient_and_sensor_source,
        ),
    )
