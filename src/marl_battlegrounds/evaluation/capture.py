"""Single-transfer host capture and lossless core-fact normalization."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from marl_battlegrounds.core.types import (
    CONTEXT_FEATURES,
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
    TransitionFacts,
)
from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    ActionMaskV1,
    AuraTransitionFactsV1,
    BaseObservationV1,
    CombatTransitionFactsV1,
    DeathTransitionFactsV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
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
    TransitionFactsV1,
)
from marl_battlegrounds.evaluation.validation import (
    _validate_frame_information_regime,  # pyright: ignore[reportPrivateUsage]
    validate_evaluation_transition_unit_v1,
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
    """Validate one already-transferred NumPy leaf without coercing it."""
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
    """Turn a NumPy ``tolist`` result into immutable native-Python values."""
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
) -> SpawnLifecycleObservationV1:
    _require_exact_type(
        source,
        SpawnLifecycleObservation,
        name="observation.spawn_lifecycle",
    )
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
    return SpawnLifecycleObservationV1.model_validate(payload)


def _normalize_base_observation_v1(source: Observation) -> BaseObservationV1:
    _require_exact_type(source, Observation, name="observation")
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
            (MAX_AGENT_SLOTS, CONTEXT_FEATURES),
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
        source.spawn_lifecycle
    )
    return BaseObservationV1.model_validate(payload)


def _normalize_action_mask_v1(source: ActionMask) -> ActionMaskV1:
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
):
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
    """Normalize an already-host core fact tree without performing a transfer.

    The caller must pass the exact ``TransitionFacts`` returned by an outer
    bundled :func:`jax.device_get`. Live JAX/device leaves are rejected so a
    future transition-capture function cannot accidentally perform 46 implicit
    transfers while walking this tree.
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
        }
    )


def _build_evaluation_frame_v1_from_host(
    context: EvaluationEpisodeContextV1,
    *,
    frame_index: int,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (object | None),
) -> EvaluationFrameV1:
    """Build one frame from an already-host bundle without transferring again."""
    _require_exact_type(context, EvaluationEpisodeContextV1, name="context")
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
    frame = EvaluationFrameV1.model_validate(
        {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "frame_id": f"{episode_id}:frame:{frame_index}",
            "simulator_step_count": simulator_step_count,
            "snapshot": snapshot,
            "base_observation": _normalize_base_observation_v1(observation),
            "action_mask": _normalize_action_mask_v1(action_mask),
            "shared_obs_information_availability_by_recipient_and_sensor_source": (
                availability_payload
            ),
        }
    )
    _validate_frame_information_regime(context, frame)
    return frame


def capture_initial_evaluation_frame_v1(
    context: EvaluationEpisodeContextV1,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
) -> EvaluationFrameV1:
    """Capture frame zero through exactly one bundled device-to-host transfer."""
    _require_exact_type(context, EvaluationEpisodeContextV1, name="context")
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
    return _build_evaluation_frame_v1_from_host(
        context,
        frame_index=0,
        state=host_state,
        observation=host_observation,
        action_mask=host_action_mask,
        shared_obs_information_availability_by_recipient_and_sensor_source=(
            host_availability
        ),
    )


def _normalize_reward_v1(source: Reward) -> object:
    _require_exact_type(source, Reward, name="canonical_reward")
    return _array_payload(
        source.rewards,
        name="canonical_reward.rewards",
        shape=(MAX_AGENT_SLOTS,),
        dtype=_FLOAT32_DTYPE,
        finite=True,
    )


def _normalize_done_flags_v1(source: DoneFlags) -> tuple[bool, bool]:
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


def capture_evaluation_transition_unit_v1(
    context: EvaluationEpisodeContextV1,
    start_frame: EvaluationFrameV1,
    successor_state: EnvState,
    successor_observation: Observation,
    successor_action_mask: ActionMask,
    transition_facts: TransitionFacts,
    canonical_reward: Reward,
    done_flags: DoneFlags,
    *,
    canonical_reward_by_team: object | None = None,
    successor_shared_obs_information_availability_by_recipient_and_sensor_source: (
        object | None
    ) = None,
    owning_task_end_reason: str | None = None,
) -> tuple[EvaluationTransitionV1, EvaluationFrameV1]:
    """Capture one adjacent transition unit through one bundled device transfer."""
    _require_exact_type(context, EvaluationEpisodeContextV1, name="context")
    _require_exact_type(start_frame, EvaluationFrameV1, name="start_frame")
    (
        host_successor_state,
        host_successor_observation,
        host_successor_action_mask,
        host_transition_facts,
        host_canonical_reward,
        host_done_flags,
        host_canonical_reward_by_team,
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
                canonical_reward_by_team,
                successor_shared_obs_information_availability_by_recipient_and_sensor_source,
            )
        ),
    )

    successor_frame = _build_evaluation_frame_v1_from_host(
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
    canonical_reward_by_team_payload: object | None = None
    if host_canonical_reward_by_team is not None:
        canonical_reward_by_team_payload = _array_payload(
            host_canonical_reward_by_team,
            name="canonical_reward_by_team",
            shape=(NUM_TEAMS,),
            dtype=_FLOAT32_DTYPE,
            finite=True,
        )
    terminated, truncated = _normalize_done_flags_v1(host_done_flags)
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
            "canonical_reward_by_agent": _normalize_reward_v1(host_canonical_reward),
            "canonical_reward_by_team": canonical_reward_by_team_payload,
            "terminated": terminated,
            "truncated": truncated,
            "owning_task_end_reason": owning_task_end_reason,
        }
    )
    validate_evaluation_transition_unit_v1(
        context,
        start_frame,
        transition,
        successor_frame,
    )
    return transition, successor_frame


def _reconstruct_transition_facts(  # pyright: ignore[reportUnusedFunction]
    source: TransitionFactsV1,
) -> TransitionFacts:
    """Reconstruct the exact core fact PyTree for losslessness tests."""
    _require_exact_type(source, TransitionFactsV1, name="transition_facts_v1")

    def action(model: JointActionV1) -> Action:
        return Action(
            move=jnp.asarray(model.move, dtype=jnp.int32),
            select_target=jnp.asarray(model.select_target, dtype=jnp.int32),
            use_ultimate=jnp.asarray(model.use_ultimate, dtype=jnp.int32),
        )

    acceptance = source.action_acceptance_facts
    action_acceptance_facts = ActionAcceptanceFacts(
        submitted_joint_action=action(acceptance.submitted_joint_action),
        accepted_joint_action=action(acceptance.accepted_joint_action),
        submitted_action_tuple_is_out_of_domain_by_actor=jnp.asarray(
            acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
            dtype=jnp.bool_,
        ),
        in_domain_move_action_is_rejected_by_actor=jnp.asarray(
            acceptance.in_domain_move_action_is_rejected_by_actor,
            dtype=jnp.bool_,
        ),
        in_domain_combat_action_pair_is_rejected_by_actor=jnp.asarray(
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
        basic_effect_is_activated_by_source=jnp.asarray(
            combat.basic_effect_is_activated_by_source, dtype=jnp.bool_
        ),
        ultimate_effect_is_activated_by_source=jnp.asarray(
            combat.ultimate_effect_is_activated_by_source, dtype=jnp.bool_
        ),
        combat_effect_has_recipient_by_source=jnp.asarray(
            combat.combat_effect_has_recipient_by_source, dtype=jnp.bool_
        ),
        combat_effect_recipient_global_slot_by_source=jnp.asarray(
            recipient_slots, dtype=jnp.int32
        ),
        raw_damage_output_by_source=jnp.asarray(
            combat.raw_damage_output_by_source, dtype=jnp.float32
        ),
        source_modified_damage_output_by_source=jnp.asarray(
            combat.source_modified_damage_output_by_source, dtype=jnp.float32
        ),
        recipient_damage_modifier_by_source=jnp.asarray(
            combat.recipient_damage_modifier_by_source, dtype=jnp.float32
        ),
        total_effective_damage_by_recipient=jnp.asarray(
            combat.total_effective_damage_by_recipient, dtype=jnp.float32
        ),
        raw_healing_output_by_source=jnp.asarray(
            combat.raw_healing_output_by_source, dtype=jnp.float32
        ),
        source_modified_healing_output_by_source=jnp.asarray(
            combat.source_modified_healing_output_by_source, dtype=jnp.float32
        ),
        recipient_healing_modifier_by_source=jnp.asarray(
            combat.recipient_healing_modifier_by_source, dtype=jnp.float32
        ),
        total_effective_healing_by_recipient=jnp.asarray(
            combat.total_effective_healing_by_recipient, dtype=jnp.float32
        ),
        health_after_combat_resolution_by_recipient=jnp.asarray(
            combat.health_after_combat_resolution_by_recipient, dtype=jnp.float32
        ),
        slow_is_applied_by_source_and_channel=jnp.asarray(
            combat.slow_is_applied_by_source_and_channel, dtype=jnp.bool_
        ),
        stun_is_applied_by_source_and_channel=jnp.asarray(
            combat.stun_is_applied_by_source_and_channel, dtype=jnp.bool_
        ),
        rogue_poison_anti_heal_is_applied_by_source=jnp.asarray(
            combat.rogue_poison_anti_heal_is_applied_by_source, dtype=jnp.bool_
        ),
        mage_burst_damage_amplification_is_applied_by_source=jnp.asarray(
            combat.mage_burst_damage_amplification_is_applied_by_source,
            dtype=jnp.bool_,
        ),
        priest_blessing_of_freedom_is_applied_by_source=jnp.asarray(
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
    return TransitionFacts(
        has_transition=jnp.asarray(source.has_transition, dtype=jnp.bool_),
        transition_start_step_count=jnp.asarray(
            source.transition_start_step_count, dtype=jnp.int32
        ),
        action_acceptance_facts=action_acceptance_facts,
        combat_transition_facts=combat_transition_facts,
        death_facts=DeathTransitionFacts(
            jnp.asarray(death.is_newly_dead_by_recipient, dtype=jnp.bool_),
            jnp.asarray(death.contributed_to_new_death_by_source, dtype=jnp.bool_),
            jnp.asarray(death.attributed_death_damage_by_source, dtype=jnp.float32),
        ),
        spawn_shield_facts=SpawnShieldTransitionFacts(
            jnp.asarray(
                shield.was_active_at_transition_start_by_agent, dtype=jnp.bool_
            ),
            jnp.asarray(shield.expired_at_transition_end_by_agent, dtype=jnp.bool_),
        ),
        respawn_facts=RespawnTransitionFacts(
            jnp.asarray(
                respawn.respawn_wave_occurred_this_transition_by_team,
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                respawn.was_respawned_this_transition_by_agent,
                dtype=jnp.bool_,
            ),
        ),
        regeneration_facts=RegenerationTransitionFacts(
            jnp.asarray(
                regeneration.combat_countdown_was_reset_by_agent, dtype=jnp.bool_
            ),
            jnp.asarray(
                regeneration.actual_health_regenerated_this_step_by_agent,
                dtype=jnp.float32,
            ),
        ),
        physical_facts=PhysicalTransitionFacts(
            jnp.asarray(physical.charge_phase_displacement_by_agent, dtype=jnp.float32),
            jnp.asarray(
                physical.ordinary_movement_phase_displacement_by_agent,
                dtype=jnp.float32,
            ),
        ),
        aura_facts=AuraTransitionFacts(
            jnp.asarray(
                aura.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                aura.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
                dtype=jnp.bool_,
            ),
        ),
        status_lifecycle_facts=StatusLifecycleTransitionFacts(
            jnp.asarray(
                lifecycle.aged_to_zero_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                lifecycle.refreshed_or_extended_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                lifecycle.broken_by_damage_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                lifecycle.cleared_by_new_death_by_recipient_and_status_channel,
                dtype=jnp.bool_,
            ),
        ),
    )


__all__ = [
    "capture_evaluation_transition_unit_v1",
    "capture_initial_evaluation_frame_v1",
    "normalize_transition_facts_v1",
]
