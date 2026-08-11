# pyright: reportPrivateUsage=false
"""Host-boundary tests for CP2 evaluation frame and fact capture."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from tests.evaluation_fixtures import evaluation_context, evaluation_env_config

import marl_battlegrounds.evaluation.capture as capture_module
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    NUM_MOVE_ACTIONS,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
    TransitionFacts,
)
from marl_battlegrounds.evaluation.capture import (
    _reconstruct_transition_facts,
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
    normalize_transition_facts_v1,
)
from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    CooldownStartedEventV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    ExecutionInformationMode,
    StatusAppliedEventV1,
    TransitionFactsV1,
)
from marl_battlegrounds.evaluation.validation import (
    validate_evaluation_transition_unit_v1,
)


@dataclass(frozen=True, slots=True)
class _StepSources:
    context: EvaluationEpisodeContextV1
    config: EnvConfig
    start_state: EnvState
    start_observation: Observation
    start_action_mask: ActionMask
    reset_info: Info
    start_frame: EvaluationFrameV1
    successor_state: EnvState
    successor_observation: Observation
    reward: Reward
    done_flags: DoneFlags
    successor_action_mask: ActionMask
    step_info: Info


def _neutral_action() -> Action:
    zeros = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    return Action(move=zeros, select_target=zeros, use_ultimate=zeros)


def _valid_shared_availability(
    context: EvaluationEpisodeContextV1,
) -> jax.Array:
    availability = np.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
        dtype=np.bool_,
    )
    for recipient, recipient_row in enumerate(context.roster):
        for sensor_source, source_row in enumerate(context.roster):
            if (
                recipient != sensor_source
                and recipient_row.configured_active
                and source_row.configured_active
                and recipient_row.configured_team_id == source_row.configured_team_id
            ):
                availability[recipient, sensor_source] = True
    return jnp.asarray(availability, dtype=jnp.bool_)


def _step_sources(
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    *,
    action: Action | None = None,
) -> _StepSources:
    config = evaluation_env_config()
    context = evaluation_context(execution_information_mode=execution_information_mode)
    start_state, start_observation, start_action_mask, reset_info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    availability = (
        _valid_shared_availability(context)
        if execution_information_mode == "shared_obs"
        else None
    )
    start_frame = capture_initial_evaluation_frame_v1(
        context,
        start_state,
        start_observation,
        start_action_mask,
        availability,
    )
    (
        successor_state,
        successor_observation,
        reward,
        done_flags,
        successor_action_mask,
        step_info,
    ) = step(
        config,
        start_state,
        start_action_mask,
        _neutral_action() if action is None else action,
        jax.random.PRNGKey(1),
    )
    return _StepSources(
        context=context,
        config=config,
        start_state=start_state,
        start_observation=start_observation,
        start_action_mask=start_action_mask,
        reset_info=reset_info,
        start_frame=start_frame,
        successor_state=successor_state,
        successor_observation=successor_observation,
        reward=reward,
        done_flags=done_flags,
        successor_action_mask=successor_action_mask,
        step_info=step_info,
    )


def _host_facts(*, initialization: bool = False) -> TransitionFacts:
    sources = _step_sources()
    facts = (
        sources.reset_info.transition_facts
        if initialization
        else sources.step_info.transition_facts
    )
    return cast(TransitionFacts, jax.device_get(facts))


def _assert_array_payload_equal(source: object, payload: object) -> None:
    source_array = cast(NDArray[np.generic], source)
    payload_array = np.asarray(payload)
    assert payload_array.shape == source_array.shape
    np.testing.assert_array_equal(payload_array, source_array)


def _assert_frame_copies_every_dynamic_leaf(
    host_state: EnvState,
    host_observation: Observation,
    host_action_mask: ActionMask,
    frame: EvaluationFrameV1,
) -> None:
    snapshot_fields = set(type(frame.snapshot).model_fields)
    assert snapshot_fields == {
        "schema_id",
        "schema_version",
        *(field for field in EnvState._fields if field != "step_count"),
    }
    assert frame.simulator_step_count == int(host_state.step_count)
    for field_name in EnvState._fields:
        if field_name == "step_count":
            continue
        _assert_array_payload_equal(
            getattr(host_state, field_name),
            getattr(frame.snapshot, field_name),
        )

    for field_name in Observation._fields:
        source_value = getattr(host_observation, field_name)
        model_value = getattr(frame.base_observation, field_name)
        if field_name in ("previous_timestep_actions", "spawn_lifecycle"):
            for nested_name in source_value._fields:
                _assert_array_payload_equal(
                    getattr(source_value, nested_name),
                    getattr(model_value, nested_name),
                )
        else:
            _assert_array_payload_equal(source_value, model_value)

    for field_name in ActionMask._fields:
        _assert_array_payload_equal(
            getattr(host_action_mask, field_name),
            getattr(frame.action_mask, field_name),
        )


def _replace_submitted_move(
    facts: TransitionFacts,
    values: NDArray[np.int32] | NDArray[np.int64] | tuple[int, ...],
) -> TransitionFacts:
    acceptance = facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action._replace(move=values)
    return facts._replace(
        action_acceptance_facts=acceptance._replace(submitted_joint_action=submitted)
    )


def test_initial_capture_performs_one_complete_transfer_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evaluation_env_config()
    context = evaluation_context()
    state, observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    original_device_get = capture_module.jax.device_get
    transferred_bundles: list[object] = []

    def spy_device_get(source: object) -> object:
        transferred_bundles.append(source)
        return original_device_get(source)

    monkeypatch.setattr(capture_module.jax, "device_get", spy_device_get)
    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
    )

    assert frame.frame_index == 0
    assert len(transferred_bundles) == 1
    bundle = cast(tuple[object, ...], transferred_bundles[0])
    assert len(bundle) == 4
    assert bundle[0] is state
    assert bundle[1] is observation
    assert bundle[2] is action_mask
    assert bundle[3] is None
    source = inspect.getsource(capture_initial_evaluation_frame_v1)
    assert "np.asarray(" not in source


def test_initial_frame_contains_full_immutable_payload_and_json_roundtrip() -> None:
    config = evaluation_env_config()
    context = evaluation_context()
    state, observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
    )
    host_state, host_observation, host_action_mask = cast(
        tuple[EnvState, Observation, ActionMask],
        jax.device_get((state, observation, action_mask)),
    )

    assert frame.episode_id == context.identity.episode_id
    assert frame.frame_id == f"{context.identity.episode_id}:frame:0"
    assert (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source is None
    )
    _assert_frame_copies_every_dynamic_leaf(
        host_state,
        host_observation,
        host_action_mask,
        frame,
    )
    assert EvaluationFrameV1.model_validate_json(frame.model_dump_json()) == frame


def test_frame_capture_enforces_both_information_regimes() -> None:
    config = evaluation_env_config()
    state, observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    no_shared_context = evaluation_context(execution_information_mode="no_shared_obs")
    shared_context = evaluation_context(execution_information_mode="shared_obs")
    availability = _valid_shared_availability(shared_context)

    no_shared_frame = capture_initial_evaluation_frame_v1(
        no_shared_context,
        state,
        observation,
        action_mask,
    )
    shared_frame = capture_initial_evaluation_frame_v1(
        shared_context,
        state,
        observation,
        action_mask,
        availability,
    )
    assert (
        no_shared_frame.shared_obs_information_availability_by_recipient_and_sensor_source
        is None
    )
    assert (
        shared_frame.shared_obs_information_availability_by_recipient_and_sensor_source
        == tuple(tuple(bool(value) for value in row) for row in availability.tolist())
    )

    with pytest.raises(ValueError, match=r"forbid|omit"):
        capture_initial_evaluation_frame_v1(
            no_shared_context,
            state,
            observation,
            action_mask,
            availability,
        )
    with pytest.raises(ValueError, match="require"):
        capture_initial_evaluation_frame_v1(
            shared_context,
            state,
            observation,
            action_mask,
        )


@pytest.mark.parametrize("forbidden_cell", ((0, 0), (0, 5), (3, 0), (0, 3)))
def test_shared_availability_rejects_every_forbidden_axis_class(
    forbidden_cell: tuple[int, int],
) -> None:
    config = evaluation_env_config()
    context = evaluation_context(execution_information_mode="shared_obs")
    state, observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    availability = (
        jnp.zeros(
            (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
            dtype=jnp.bool_,
        )
        .at[forbidden_cell]
        .set(True)
    )

    with pytest.raises(ValueError, match="diagonal, cross-team"):
        capture_initial_evaluation_frame_v1(
            context,
            state,
            observation,
            action_mask,
            availability,
        )


@pytest.mark.parametrize(
    ("invalid_source", "expected_error"),
    (
        ("state_shape", ValueError),
        ("observation_dtype", TypeError),
        ("observation_nonfinite", ValueError),
        ("state_category", ValueError),
        ("mask_dtype", TypeError),
    ),
)
def test_frame_capture_strictly_rejects_malformed_sources(
    invalid_source: str,
    expected_error: type[Exception],
) -> None:
    config = evaluation_env_config()
    context = evaluation_context()
    state, observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    if invalid_source == "state_shape":
        state = state._replace(
            agent_positions=jnp.zeros((MAX_AGENT_SLOTS - 1, 2), dtype=jnp.float32)
        )
    elif invalid_source == "observation_dtype":
        observation = observation._replace(
            self_features=observation.self_features.astype(jnp.int32)
        )
    elif invalid_source == "observation_nonfinite":
        observation = observation._replace(
            self_features=observation.self_features.at[0, 0].set(jnp.nan)
        )
    elif invalid_source == "state_category":
        state = state._replace(
            previous_timestep_move_actions=(
                state.previous_timestep_move_actions.at[0].set(NUM_MOVE_ACTIONS)
            )
        )
    else:
        action_mask = action_mask._replace(
            move_mask=action_mask.move_mask.astype(jnp.int32)
        )

    with pytest.raises(expected_error):
        capture_initial_evaluation_frame_v1(
            context,
            state,
            observation,
            action_mask,
        )


def test_core_does_no_evaluation_work_when_capture_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfers = 0

    def forbidden_device_get(_source: object) -> object:
        nonlocal transfers
        transfers += 1
        raise AssertionError("core reset/step must not invoke evaluation capture")

    monkeypatch.setattr(capture_module.jax, "device_get", forbidden_device_get)
    config = evaluation_env_config()
    state, _observation, action_mask, _info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    step(
        config,
        state,
        action_mask,
        _neutral_action(),
        jax.random.PRNGKey(1),
    )
    assert transfers == 0


def test_fact_normalization_accounts_for_all_46_leaves_and_is_lossless() -> None:
    host_facts = _host_facts()
    leaves = jax.tree_util.tree_leaves(host_facts)
    assert len(leaves) == 46
    assert all(type(leaf) is np.ndarray for leaf in leaves)
    assert sum(cast(NDArray[np.generic], leaf).nbytes for leaf in leaves) == 1657

    normalized = normalize_transition_facts_v1(host_facts)
    json_roundtrip = TransitionFactsV1.model_validate_json(normalized.model_dump_json())
    assert json_roundtrip == normalized
    reconstructed = cast(
        TransitionFacts,
        jax.device_get(_reconstruct_transition_facts(json_roundtrip)),
    )
    assert jax.tree_util.tree_structure(reconstructed) == jax.tree_util.tree_structure(
        host_facts
    )
    for expected, actual in zip(
        jax.tree_util.tree_leaves(host_facts),
        jax.tree_util.tree_leaves(reconstructed),
        strict=True,
    ):
        expected_array = cast(NDArray[np.generic], expected)
        actual_array = cast(NDArray[np.generic], actual)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def test_initialization_facts_normalize_but_cannot_construct_transition() -> None:
    initialization_facts = normalize_transition_facts_v1(
        _host_facts(initialization=True)
    )
    assert initialization_facts.has_transition is False
    assert initialization_facts.transition_start_step_count == -1
    assert (
        TransitionFactsV1.model_validate_json(initialization_facts.model_dump_json())
        == initialization_facts
    )

    with pytest.raises(ValidationError, match="has_transition"):
        EvaluationTransitionV1(
            episode_id="episode-001",
            transition_index=0,
            transition_id="episode-001:transition:0",
            start_frame_id="episode-001:frame:0",
            successor_frame_id="episode-001:frame:1",
            facts=initialization_facts,
            events=(),
            canonical_reward_by_agent=(0.0,) * MAX_AGENT_SLOTS,
            canonical_reward_by_team=None,
            terminated=False,
            truncated=False,
            owning_task_end_reason=None,
        )


def test_fact_normalization_is_host_only_and_performs_no_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _step_sources()
    transfers = 0

    def forbidden_device_get(_source: object) -> object:
        nonlocal transfers
        transfers += 1
        raise AssertionError("fact normalization must never transfer")

    monkeypatch.setattr(capture_module.jax, "device_get", forbidden_device_get)
    with pytest.raises(TypeError, match="still a JAX/device array"):
        normalize_transition_facts_v1(sources.step_info.transition_facts)
    assert transfers == 0


def test_submitted_actions_are_int32_and_accepted_actions_are_bounded() -> None:
    host_facts = _host_facts()
    submitted_values = np.zeros((MAX_AGENT_SLOTS,), dtype=np.int32)
    submitted_values[0] = np.iinfo(np.int32).min
    submitted_values[1] = np.iinfo(np.int32).max
    submitted_facts = _replace_submitted_move(host_facts, submitted_values)

    normalized = normalize_transition_facts_v1(submitted_facts)
    assert normalized.action_acceptance_facts.submitted_joint_action.move[:2] == (
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    )
    reconstructed = cast(
        TransitionFacts,
        jax.device_get(_reconstruct_transition_facts(normalized)),
    )
    np.testing.assert_array_equal(
        reconstructed.action_acceptance_facts.submitted_joint_action.move,
        submitted_values,
    )

    acceptance = host_facts.action_acceptance_facts
    invalid_accepted_move = acceptance.accepted_joint_action.move.copy()
    invalid_accepted_move[0] = NUM_MOVE_ACTIONS
    invalid_facts = host_facts._replace(
        action_acceptance_facts=acceptance._replace(
            accepted_joint_action=acceptance.accepted_joint_action._replace(
                move=invalid_accepted_move
            )
        )
    )
    with pytest.raises(ValueError, match="out-of-domain category"):
        normalize_transition_facts_v1(invalid_facts)


def test_recipient_sentinel_normalization_is_consistent_and_reversible() -> None:
    host_facts = _host_facts()
    has_recipient = np.zeros((MAX_AGENT_SLOTS,), dtype=np.bool_)
    recipient_slots = np.full((MAX_AGENT_SLOTS,), -1, dtype=np.int32)
    has_recipient[0] = True
    recipient_slots[0] = 5
    combat = host_facts.combat_transition_facts._replace(
        combat_effect_has_recipient_by_source=has_recipient,
        combat_effect_recipient_global_slot_by_source=recipient_slots,
    )
    routed_facts = host_facts._replace(combat_transition_facts=combat)

    normalized = normalize_transition_facts_v1(routed_facts)
    normalized_recipients = (
        normalized.combat_transition_facts.combat_effect_recipient_global_slot_by_source
    )
    assert normalized_recipients == (
        5,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    reconstructed = cast(
        TransitionFacts,
        jax.device_get(_reconstruct_transition_facts(normalized)),
    )
    np.testing.assert_array_equal(
        reconstructed.combat_transition_facts.combat_effect_recipient_global_slot_by_source,
        recipient_slots,
    )


@pytest.mark.parametrize(
    ("has_recipient", "recipient", "message"),
    ((False, 5, "sentinel -1"), (True, -1, "must be in")),
)
def test_fact_normalization_rejects_recipient_presence_disagreement(
    has_recipient: bool,
    recipient: int,
    message: str,
) -> None:
    host_facts = _host_facts()
    has_values = np.zeros((MAX_AGENT_SLOTS,), dtype=np.bool_)
    recipient_values = np.full((MAX_AGENT_SLOTS,), -1, dtype=np.int32)
    has_values[0] = has_recipient
    recipient_values[0] = recipient
    combat = host_facts.combat_transition_facts._replace(
        combat_effect_has_recipient_by_source=has_values,
        combat_effect_recipient_global_slot_by_source=recipient_values,
    )

    with pytest.raises(ValueError, match=message):
        normalize_transition_facts_v1(
            host_facts._replace(combat_transition_facts=combat)
        )


@pytest.mark.parametrize(
    ("invalid_source", "expected_error"),
    (
        ("list_leaf", TypeError),
        ("wrong_shape", ValueError),
        ("wrong_dtype", TypeError),
        ("nonfinite", ValueError),
        ("false_wrong_sentinel", ValueError),
    ),
)
def test_fact_normalization_strictly_rejects_malformed_host_sources(
    invalid_source: str,
    expected_error: type[Exception],
) -> None:
    facts = _host_facts()
    if invalid_source == "list_leaf":
        facts = _replace_submitted_move(facts, (0,) * MAX_AGENT_SLOTS)
    elif invalid_source == "wrong_shape":
        death = facts.death_facts._replace(
            is_newly_dead_by_recipient=np.zeros((MAX_AGENT_SLOTS - 1,), dtype=np.bool_)
        )
        facts = facts._replace(death_facts=death)
    elif invalid_source == "wrong_dtype":
        facts = _replace_submitted_move(
            facts,
            np.zeros((MAX_AGENT_SLOTS,), dtype=np.int64),
        )
    elif invalid_source == "nonfinite":
        combat = facts.combat_transition_facts._replace(
            raw_damage_output_by_source=np.full(
                (MAX_AGENT_SLOTS,), np.nan, dtype=np.float32
            )
        )
        facts = facts._replace(combat_transition_facts=combat)
    else:
        facts = _host_facts(initialization=True)._replace(
            transition_start_step_count=np.asarray(0, dtype=np.int32)
        )

    with pytest.raises(expected_error):
        normalize_transition_facts_v1(facts)


def test_transition_capture_uses_one_transfer_and_validates_real_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _step_sources()
    original_device_get = capture_module.jax.device_get
    original_validator = capture_module.validate_evaluation_transition_unit_v1
    transferred_bundles: list[object] = []
    validated_units: list[
        tuple[
            EvaluationEpisodeContextV1,
            EvaluationFrameV1,
            EvaluationTransitionV1,
            EvaluationFrameV1,
        ]
    ] = []

    def spy_device_get(source: object) -> object:
        transferred_bundles.append(source)
        return original_device_get(source)

    def spy_validator(
        context: EvaluationEpisodeContextV1,
        start_frame: EvaluationFrameV1,
        transition: EvaluationTransitionV1,
        successor_frame: EvaluationFrameV1,
    ) -> None:
        validated_units.append((context, start_frame, transition, successor_frame))
        original_validator(context, start_frame, transition, successor_frame)

    monkeypatch.setattr(capture_module.jax, "device_get", spy_device_get)
    monkeypatch.setattr(
        capture_module,
        "validate_evaluation_transition_unit_v1",
        spy_validator,
    )
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
    )

    assert len(transferred_bundles) == 1
    bundle = cast(tuple[object, ...], transferred_bundles[0])
    assert len(bundle) == 8
    assert bundle[0] is sources.successor_state
    assert bundle[1] is sources.successor_observation
    assert bundle[2] is sources.successor_action_mask
    assert bundle[3] is sources.step_info.transition_facts
    assert bundle[4] is sources.reward
    assert bundle[5] is sources.done_flags
    assert bundle[6:] == (None, None)
    assert validated_units == (
        [
            (
                sources.context,
                sources.start_frame,
                transition,
                successor_frame,
            )
        ]
    )
    assert transition.transition_index == sources.start_frame.frame_index
    assert transition.transition_id == "episode-001:transition:0"
    assert transition.start_frame_id == sources.start_frame.frame_id
    assert transition.successor_frame_id == successor_frame.frame_id
    assert transition.canonical_reward_by_team is None
    assert transition.owning_task_end_reason is None
    assert successor_frame.frame_index == sources.start_frame.frame_index + 1
    assert (
        successor_frame.simulator_step_count
        == sources.start_frame.simulator_step_count + 1
    )
    assert (
        EvaluationTransitionV1.model_validate_json(transition.model_dump_json())
        == transition
    )


def test_event_bearing_public_trajectory_preserves_successor_policy_inputs() -> None:
    zeros = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    mage_target_none_ultimate = Action(
        move=zeros,
        select_target=zeros,
        use_ultimate=zeros.at[0].set(1),
    )
    sources = _step_sources(action=mage_target_none_ultimate)

    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
    )

    assert tuple(event.event_type for event in transition.events) == (
        "ability_activated",
        "cooldown_started",
        "status_applied",
    )
    ability, cooldown, status = transition.events
    assert isinstance(ability, AbilityActivatedEventV1)
    assert ability.source_global_slot == 0
    assert ability.ability_component == "ultimate"
    assert ability.recipient_global_slot is None
    assert isinstance(cooldown, CooldownStartedEventV1)
    assert cooldown.agent_global_slot == 0
    assert isinstance(status, StatusAppliedEventV1)
    assert status.source_global_slot == 0
    assert status.recipient_global_slot == 0
    assert status.status_channel == 7
    assert status.status_id == "mage_burst_damage_amplification"
    accepted = transition.facts.action_acceptance_facts.accepted_joint_action
    assert accepted.select_target[0] == 0
    assert accepted.use_ultimate[0] == 1

    host_successor_state, host_successor_observation, host_successor_action_mask = cast(
        tuple[EnvState, Observation, ActionMask],
        jax.device_get(
            (
                sources.successor_state,
                sources.successor_observation,
                sources.successor_action_mask,
            )
        ),
    )
    _assert_frame_copies_every_dynamic_leaf(
        host_successor_state,
        host_successor_observation,
        host_successor_action_mask,
        successor_frame,
    )


def test_four_record_validation_survives_json_and_rejects_cross_record_drift() -> None:
    sources = _step_sources()
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
    )
    context_roundtrip = EvaluationEpisodeContextV1.model_validate_json(
        sources.context.model_dump_json()
    )
    start_roundtrip = EvaluationFrameV1.model_validate_json(
        sources.start_frame.model_dump_json()
    )
    transition_roundtrip = EvaluationTransitionV1.model_validate_json(
        transition.model_dump_json()
    )
    successor_roundtrip = EvaluationFrameV1.model_validate_json(
        successor_frame.model_dump_json()
    )
    validate_evaluation_transition_unit_v1(
        context_roundtrip,
        start_roundtrip,
        transition_roundtrip,
        successor_roundtrip,
    )

    wrong_successor_index = successor_frame.frame_index + 1
    wrong_successor = successor_frame.model_copy(
        update={
            "frame_index": wrong_successor_index,
            "frame_id": (f"{successor_frame.episode_id}:frame:{wrong_successor_index}"),
        }
    )
    with pytest.raises(ValueError, match="successor frame index"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition,
            wrong_successor,
        )

    wrong_epoch = successor_frame.model_copy(
        update={
            "simulator_step_count": successor_frame.simulator_step_count + 1,
        }
    )
    with pytest.raises(ValueError, match="successor simulator step"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition,
            wrong_epoch,
        )

    wrong_facts = transition.facts.model_copy(
        update={
            "transition_start_step_count": (
                transition.facts.transition_start_step_count + 1
            )
        }
    )
    with pytest.raises(ValueError, match="facts must name"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(update={"facts": wrong_facts}),
            successor_frame,
        )

    wrong_identity = sources.context.identity.model_copy(
        update={"episode_id": "different-episode"}
    )
    with pytest.raises(ValueError, match="context episode"):
        validate_evaluation_transition_unit_v1(
            sources.context.model_copy(update={"identity": wrong_identity}),
            sources.start_frame,
            transition,
            successor_frame,
        )


def test_four_record_validation_rejects_event_omission_order_and_payload_drift() -> (
    None
):
    zeros = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    sources = _step_sources(
        action=Action(
            move=zeros,
            select_target=zeros,
            use_ultimate=zeros.at[0].set(1),
        )
    )
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
    )
    assert len(transition.events) == 3

    with pytest.raises(ValueError, match="exactly equal canonical fact decoding"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(update={"events": transition.events[:-1]}),
            successor_frame,
        )

    with pytest.raises(ValueError, match="structural revalidation"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(
                update={"events": tuple(reversed(transition.events))}
            ),
            successor_frame,
        )

    status = cast(StatusAppliedEventV1, transition.events[-1])
    wrong_status = status.model_copy(update={"source_global_slot": 1})
    with pytest.raises(ValueError, match="exactly equal canonical fact decoding"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(
                update={"events": (*transition.events[:-1], wrong_status)}
            ),
            successor_frame,
        )


def test_four_record_validation_rejects_inactive_dynamic_padding_drift() -> None:
    sources = _step_sources()
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
    )
    inactive_slot = 3

    displacement_rows = list(
        transition.facts.physical_facts.ordinary_movement_phase_displacement_by_agent
    )
    displacement_rows[inactive_slot] = (1.0, 0.0)
    physical = transition.facts.physical_facts.model_copy(
        update={
            "ordinary_movement_phase_displacement_by_agent": tuple(displacement_rows)
        }
    )
    displacement_facts = transition.facts.model_copy(
        update={"physical_facts": physical}
    )
    displacement_events = decode_evaluation_events_v1(
        sources.context,
        sources.start_frame,
        displacement_facts,
        successor_frame,
    )
    with pytest.raises(ValueError, match=r"inactive slot 3.*physical_facts"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(
                update={
                    "facts": displacement_facts,
                    "events": displacement_events,
                }
            ),
            successor_frame,
        )

    accepted_move = list(
        transition.facts.action_acceptance_facts.accepted_joint_action.move
    )
    accepted_move[inactive_slot] = 1
    accepted_action = (
        transition.facts.action_acceptance_facts.accepted_joint_action.model_copy(
            update={"move": tuple(accepted_move)}
        )
    )
    acceptance = transition.facts.action_acceptance_facts.model_copy(
        update={"accepted_joint_action": accepted_action}
    )
    accepted_facts = transition.facts.model_copy(
        update={"action_acceptance_facts": acceptance}
    )
    with pytest.raises(
        ValueError,
        match=r"inactive slot 3.*accepted_joint_action",
    ):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(update={"facts": accepted_facts}),
            successor_frame,
        )

    aura_facts_model = transition.facts.aura_facts
    mage_aura_facts = (
        aura_facts_model.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary
    )
    mage_coverage = [list(row) for row in mage_aura_facts]
    mage_coverage[0][inactive_slot] = True
    aura = transition.facts.aura_facts.model_copy(
        update={
            "is_covered_by_mage_damage_aura_by_emitter_and_beneficiary": tuple(
                tuple(row) for row in mage_coverage
            )
        }
    )
    aura_facts = transition.facts.model_copy(update={"aura_facts": aura})
    with pytest.raises(ValueError, match=r"inactive slot 3.*beneficiary column"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(update={"facts": aura_facts}),
            successor_frame,
        )

    recipient_slots = list(
        transition.facts.combat_transition_facts.combat_effect_recipient_global_slot_by_source
    )
    has_recipient = list(
        transition.facts.combat_transition_facts.combat_effect_has_recipient_by_source
    )
    recipient_slots[0] = inactive_slot
    has_recipient[0] = True
    combat = transition.facts.combat_transition_facts.model_copy(
        update={
            "combat_effect_has_recipient_by_source": tuple(has_recipient),
            "combat_effect_recipient_global_slot_by_source": tuple(recipient_slots),
        }
    )
    routed_facts = transition.facts.model_copy(
        update={"combat_transition_facts": combat}
    )
    with pytest.raises(ValueError, match=r"recipient routes.*inactive"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            transition.model_copy(update={"facts": routed_facts}),
            successor_frame,
        )

    positions = list(sources.start_frame.snapshot.agent_positions)
    positions[inactive_slot] = (1.0, 0.0)
    snapshot = sources.start_frame.snapshot.model_copy(
        update={"agent_positions": tuple(positions)}
    )
    with pytest.raises(ValueError, match=r"inactive slot 3.*agent_positions"):
        validate_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame.model_copy(update={"snapshot": snapshot}),
            transition,
            successor_frame,
        )


def test_transition_capture_preserves_shared_availability_and_team_reward() -> None:
    sources = _step_sources("shared_obs")
    availability = _valid_shared_availability(sources.context)
    team_reward = jnp.asarray((1.25, -1.25), dtype=jnp.float32)
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        sources.context,
        sources.start_frame,
        sources.successor_state,
        sources.successor_observation,
        sources.successor_action_mask,
        sources.step_info.transition_facts,
        sources.reward,
        sources.done_flags,
        canonical_reward_by_team=team_reward,
        successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
            availability
        ),
    )

    assert transition.canonical_reward_by_team == (1.25, -1.25)
    assert (
        successor_frame.shared_obs_information_availability_by_recipient_and_sensor_source
        == tuple(tuple(bool(value) for value in row) for row in availability.tolist())
    )

    with pytest.raises(ValueError, match="require"):
        capture_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            sources.successor_state,
            sources.successor_observation,
            sources.successor_action_mask,
            sources.step_info.transition_facts,
            sources.reward,
            sources.done_flags,
        )


def test_transition_capture_rejects_initialization_facts_and_malformed_rewards() -> (
    None
):
    sources = _step_sources()
    with pytest.raises(ValueError, match="initialization facts"):
        capture_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            sources.successor_state,
            sources.successor_observation,
            sources.successor_action_mask,
            sources.reset_info.transition_facts,
            sources.reward,
            sources.done_flags,
        )

    with pytest.raises(TypeError, match=r"canonical_reward\.rewards"):
        capture_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            sources.successor_state,
            sources.successor_observation,
            sources.successor_action_mask,
            sources.step_info.transition_facts,
            Reward(rewards=sources.reward.rewards.astype(jnp.int32)),
            sources.done_flags,
        )

    with pytest.raises(TypeError, match="canonical_reward_by_team"):
        capture_evaluation_transition_unit_v1(
            sources.context,
            sources.start_frame,
            sources.successor_state,
            sources.successor_observation,
            sources.successor_action_mask,
            sources.step_info.transition_facts,
            sources.reward,
            sources.done_flags,
            canonical_reward_by_team=jnp.zeros((2,), dtype=jnp.int32),
        )
