"""Reusable strict host-model fixtures for evaluation tests."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_TDM_ALLY_SCORE,
    CONTEXT_FEATURE_TDM_ENEMY_SCORE,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_NEUTRAL,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_TDM,
    WARRIOR_CLASS_ID,
    Action,
    DoneFlags,
    EnvConfig,
    Reward,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.catalog import (
    build_code_revision_v1,
    build_evaluation_episode_context_v1,
    build_evaluation_seed_protocol_v1,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV1,
    CaptureProfile,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    EvaluationFrameV1,
    EvaluationRole,
    EvaluationSeedProtocolV1,
    EvaluationTransitionV1,
    ExecutionInformationMode,
    NotApplicablePolicySlotV1,
    PolicyAssignmentSlotV1,
    VersionedIdentityV1,
)

_DIGEST_A = "1" * 64
_DIGEST_B = "2" * 64
_DIGEST_C = "3" * 64


def evaluation_env_config(
    *,
    team_sizes: tuple[int, int] = (3, 2),
    task_mode: int = TASK_MODE_NEUTRAL,
    team_deathmatch_score_threshold: int = 0,
    max_steps: int = 100,
) -> EnvConfig:
    """Return one asymmetric, padded, duplicate-class-valid configuration."""
    requested_classes = jnp.asarray(
        (
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            PRIEST_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            PRIEST_CLASS_ID,
        ),
        dtype=jnp.int32,
    )
    profile = resolve_agent_profile(
        requested_classes,
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    y_coordinates = jnp.linspace(
        1.5,
        10.5,
        MAX_AGENTS_PER_TEAM,
        dtype=jnp.float32,
    )
    spawn_pads = jnp.stack(
        (
            jnp.stack((jnp.full_like(y_coordinates, 1.5), y_coordinates), axis=-1),
            jnp.stack(
                (jnp.full_like(y_coordinates, 18.5), y_coordinates),
                axis=-1,
            ),
        ),
        axis=0,
    )
    return EnvConfig(
        task_mode=task_mode,
        team_deathmatch_score_threshold=team_deathmatch_score_threshold,
        max_steps=max_steps,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros(
            (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=spawn_pads,
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 7), dtype=jnp.int32),
    )


def content_identity(
    name: str,
    *,
    digest: str = _DIGEST_A,
) -> ContentAddressedIdentityV1:
    """Return one deterministic content-addressed identity."""
    return ContentAddressedIdentityV1(
        identifier=name,
        version=1,
        canonical_digest=digest,
    )


def evaluation_episode_identity(
    *,
    with_scenario: bool = False,
    episode_id: str = "episode-001",
) -> EvaluationEpisodeIdentityV1:
    """Return deterministic runner-owned episode identity."""
    return EvaluationEpisodeIdentityV1(
        run_id="run-001",
        evaluation_id="evaluation-001",
        matchup_id="matchup-001",
        match_id="match-001",
        episode_id=episode_id,
        paired_comparison_key="pair-001",
        evaluation_suite=content_identity("suite", digest=_DIGEST_A),
        experiment_manifest=content_identity("manifest", digest=_DIGEST_B),
        task=content_identity("task", digest=_DIGEST_C),
        layout=content_identity("layout", digest=_DIGEST_A),
        curriculum=None,
        scenario=(
            content_identity("scenario", digest=_DIGEST_B) if with_scenario else None
        ),
    )


def policy_assignments(
    config: EnvConfig,
) -> tuple[PolicyAssignmentSlotV1, ...]:
    """Return exactly ten policy rows aligned with the resolved active roster."""
    active = tuple(
        bool(value) for value in np.asarray(config.agent_profile.active_mask)
    )
    rows: list[PolicyAssignmentSlotV1] = []
    for slot in range(MAX_AGENT_SLOTS):
        if not active[slot]:
            rows.append(NotApplicablePolicySlotV1(global_slot=slot))
            continue
        if slot == 0:
            role: EvaluationRole = "focal"
        elif slot < MAX_AGENTS_PER_TEAM:
            role = "cooperative_partner"
        else:
            role = "adversarial_opponent"
        rows.append(
            AssignedPolicySlotV1(
                global_slot=slot,
                evaluation_role=role,
                policy_kind="checkpoint",
                policy_id=f"policy-{slot}",
                policy_content_digest=_DIGEST_A,
                checkpoint_digest=_DIGEST_B,
                algorithm_id="mappo",
                training_run_id=f"training-run-{slot}",
                training_step=10_000,
                population_member_id=f"population-member-{slot}",
                parameter_sharing_group_id=(
                    "team-a" if slot < MAX_AGENTS_PER_TEAM else "team-b"
                ),
                preprocessing=VersionedIdentityV1(
                    identifier="base-observation",
                    version=1,
                ),
                normalization=VersionedIdentityV1(
                    identifier="none",
                    version=1,
                ),
                execution_mode="deterministic",
            )
        )
    return tuple(rows)


def evaluation_seed_protocol(
    *,
    with_scenario: bool = False,
) -> EvaluationSeedProtocolV1:
    """Return named realized seeds consistent with the policy fixture."""
    return build_evaluation_seed_protocol_v1(
        seed_protocol=VersionedIdentityV1(identifier="split-v1", version=1),
        root_seed=1,
        episode_seed=2,
        layout_seed=3,
        environment_seed=4,
        focal_policy_seed=5,
        evaluation_seed=6,
        cooperative_partner_seed=7,
        adversarial_opponent_seed=8,
        scenario_seed=9 if with_scenario else "not_applicable",
    )


def evaluation_context(
    *,
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    actor_projection: VersionedIdentityV1 | None = None,
    with_scenario: bool = False,
    public_agent_id_prefix: str = "agent-slot",
    capture_profile: CaptureProfile | None = None,
    expected_horizon: int = 100,
    aggregation_keys: tuple[AggregationKeyV1, ...] | None = None,
    episode_id: str = "episode-001",
    config: EnvConfig | None = None,
) -> EvaluationEpisodeContextV1:
    """Build a complete valid episode context through the public constructor."""
    resolved_config = evaluation_env_config() if config is None else config
    active_mask = tuple(
        bool(value) for value in np.asarray(resolved_config.agent_profile.active_mask)
    )
    seed_protocol = evaluation_seed_protocol(with_scenario=with_scenario)
    seed_updates: dict[str, int | str] = {}
    if sum(active_mask[:MAX_AGENTS_PER_TEAM]) <= 1:
        seed_updates["cooperative_partner_seed"] = "not_applicable"
    if not any(active_mask[MAX_AGENTS_PER_TEAM:]):
        seed_updates["adversarial_opponent_seed"] = "not_applicable"
    if seed_updates:
        seed_protocol = seed_protocol.model_copy(update=seed_updates)
    return build_evaluation_episode_context_v1(
        identity=evaluation_episode_identity(
            with_scenario=with_scenario,
            episode_id=episode_id,
        ),
        aggregation_keys=(
            aggregation_keys
            if aggregation_keys is not None
            else (
                AggregationKeyV1(
                    name="information_regime",
                    value=execution_information_mode,
                ),
                AggregationKeyV1(name="side", value="team_a"),
            )
        ),
        expected_horizon=expected_horizon,
        config=resolved_config,
        public_agent_id_by_global_slot=tuple(
            f"{public_agent_id_prefix}-{slot}" for slot in range(MAX_AGENT_SLOTS)
        ),
        policy_assignments=policy_assignments(resolved_config),
        seed_protocol=seed_protocol,
        capture_profile=(
            capture_profile
            if capture_profile is not None
            else (
                "scenario_metric_complete"
                if with_scenario
                else "evaluation_metric_complete"
            )
        ),
        execution_information_mode=execution_information_mode,
        actor_projection=(
            actor_projection
            if actor_projection is not None
            else VersionedIdentityV1(
                identifier=f"actor-projection-{execution_information_mode}",
                version=1,
            )
        ),
        critic_information_regime=VersionedIdentityV1(
            identifier="privileged-world-state",
            version=1,
        ),
        canonical_reward_mode=VersionedIdentityV1(
            identifier="canonical-task-reward",
            version=1,
        ),
        shaping_configuration=content_identity("no-shaping", digest=_DIGEST_C),
        code_revision=build_code_revision_v1(
            package_version="0.0.0",
            commit_sha="a" * 40,
            source_tree_digest=_DIGEST_A,
            is_dirty=False,
            dirty_patch_digest=None,
        ),
    )


@dataclass(frozen=True, slots=True)
class CapturedEvaluationTrajectory:
    """One public reset/step trajectory normalized through the CP2 seam."""

    context: EvaluationEpisodeContextV1
    frames: tuple[EvaluationFrameV1, ...]
    transitions: tuple[EvaluationTransitionV1, ...]


def neutral_action() -> Action:
    """Return the canonical all-Stay/no-target/no-Ultimate joint action."""
    zeros = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    return Action(move=zeros, select_target=zeros, use_ultimate=zeros)


def mage_target_none_ultimate_action() -> Action:
    """Return an action that activates slot-zero Mage Ultimate at target-none."""
    action = neutral_action()
    return Action(
        move=action.move,
        select_target=action.select_target,
        use_ultimate=action.use_ultimate.at[0].set(1),
    )


def valid_shared_availability(
    context: EvaluationEpisodeContextV1,
) -> jax.Array:
    """Return a valid same-team off-diagonal SharedObs availability matrix."""
    availability = np.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
        dtype=np.bool_,
    )
    for recipient, recipient_row in enumerate(context.roster):
        for sensor_source, source_row in enumerate(context.roster):
            availability[recipient, sensor_source] = bool(
                recipient != sensor_source
                and recipient_row.configured_active
                and source_row.configured_active
                and recipient_row.configured_team_id == source_row.configured_team_id
            )
    return jnp.asarray(availability, dtype=jnp.bool_)


def captured_evaluation_trajectory(
    *,
    transition_count: int = 2,
    capture_profile: CaptureProfile = "evaluation_metric_complete",
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    actor_projection: VersionedIdentityV1 | None = None,
    expected_horizon: int = 100,
    with_scenario: bool = False,
    episode_id: str = "episode-001",
    aggregation_keys: tuple[AggregationKeyV1, ...] | None = None,
    actions: tuple[Action, ...] | None = None,
    config: EnvConfig | None = None,
) -> CapturedEvaluationTrajectory:
    """Capture a deterministic trajectory through public reset, step, and CP2 APIs."""
    if transition_count < 0:
        raise ValueError("transition_count must be nonnegative")
    resolved_config = evaluation_env_config() if config is None else config
    if transition_count > resolved_config.max_steps:
        raise ValueError("transition_count cannot exceed the fixture config horizon")
    if actions is not None and len(actions) != transition_count:
        raise ValueError("actions must contain exactly one joint action per transition")

    context = evaluation_context(
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        with_scenario=with_scenario,
        capture_profile=capture_profile,
        expected_horizon=expected_horizon,
        aggregation_keys=aggregation_keys,
        episode_id=episode_id,
        config=resolved_config,
    )
    state, observation, action_mask, _reset_info = reset(
        resolved_config,
        jax.random.PRNGKey(0),
    )
    availability = (
        valid_shared_availability(context)
        if execution_information_mode == "shared_obs"
        else None
    )
    current_frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
        availability,
    )
    frames = [current_frame]
    transitions: list[EvaluationTransitionV1] = []
    for transition_index in range(transition_count):
        (
            state,
            observation,
            canonical_reward,
            done_flags,
            action_mask,
            info,
        ) = step(
            resolved_config,
            state,
            action_mask,
            neutral_action() if actions is None else actions[transition_index],
            jax.random.PRNGKey(transition_index + 1),
        )
        transition, current_frame = capture_evaluation_transition_unit_v1(
            context,
            current_frame,
            state,
            observation,
            action_mask,
            info.transition_facts,
            canonical_reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
                availability
            ),
        )
        transitions.append(transition)
        frames.append(current_frame)

    return CapturedEvaluationTrajectory(
        context=context,
        frames=tuple(frames),
        transitions=tuple(transitions),
    )


def captured_team_deathmatch_threshold_trajectory(
    *,
    at_horizon: bool = False,
    episode_id: str = "episode-001",
    aggregation_keys: tuple[AggregationKeyV1, ...] | None = None,
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    actor_projection: VersionedIdentityV1 | None = None,
    with_scenario: bool = False,
) -> CapturedEvaluationTrajectory:
    """Capture one authoritative Team A threshold win for host-side tests."""
    maximum_episode_steps = 1 if at_horizon else 100
    config = evaluation_env_config(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=1,
        max_steps=maximum_episode_steps,
    )
    context = evaluation_context(
        expected_horizon=maximum_episode_steps,
        episode_id=episode_id,
        aggregation_keys=aggregation_keys,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        with_scenario=with_scenario,
        config=config,
    )
    availability = (
        valid_shared_availability(context)
        if execution_information_mode == "shared_obs"
        else None
    )
    start_state, start_observation, start_action_mask, _reset_info = reset(
        config,
        jax.random.PRNGKey(0),
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
        _canonical_reward,
        horizon_done_flags,
        successor_action_mask,
        step_info,
    ) = step(
        config,
        start_state,
        start_action_mask,
        neutral_action(),
        jax.random.PRNGKey(1),
    )

    team_b_recipient = MAX_AGENTS_PER_TEAM
    successor_state = successor_state._replace(
        team_deathmatch_scores=jnp.asarray((1, 0), dtype=jnp.int32),
        alive_mask=successor_state.alive_mask.at[team_b_recipient].set(False),
        current_health=successor_state.current_health.at[team_b_recipient].set(0.0),
    )
    team_a_active = jnp.logical_and(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids == 1,
    )
    team_b_active = jnp.logical_and(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids == 2,
    )
    context_features = successor_observation.context_features
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_ALLY_SCORE].set(
        jnp.where(
            team_a_active,
            1.0,
            context_features[:, CONTEXT_FEATURE_TDM_ALLY_SCORE],
        )
    )
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_ENEMY_SCORE].set(
        jnp.where(
            team_b_active,
            1.0,
            context_features[:, CONTEXT_FEATURE_TDM_ENEMY_SCORE],
        )
    )
    successor_observation = successor_observation._replace(
        context_features=context_features
    )
    transition_facts = step_info.transition_facts._replace(
        death_facts=step_info.transition_facts.death_facts._replace(
            is_newly_dead_by_recipient=(
                step_info.transition_facts.death_facts.is_newly_dead_by_recipient.at[
                    team_b_recipient
                ].set(True)
            )
        ),
        team_deathmatch_facts=(
            step_info.transition_facts.team_deathmatch_facts._replace(
                outcome=jnp.asarray(
                    TASK_MODE_OUTCOME_TEAM_A_WIN,
                    dtype=jnp.int32,
                )
            )
        ),
    )
    reward = Reward(
        rewards=jnp.asarray(
            (1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0),
            dtype=jnp.float32,
        )
    )
    done_flags = DoneFlags(
        terminated=jnp.asarray(True, dtype=jnp.bool_),
        truncated=horizon_done_flags.truncated,
    )
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        context,
        start_frame,
        successor_state,
        successor_observation,
        successor_action_mask,
        transition_facts,
        reward,
        done_flags,
        successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
            availability
        ),
    )
    return CapturedEvaluationTrajectory(
        context=context,
        frames=(start_frame, successor_frame),
        transitions=(transition,),
    )


def captured_resumed_team_deathmatch_horizon_trajectory() -> (
    CapturedEvaluationTrajectory
):
    """Capture a resumed unequal-score TDM prefix through its horizon draw."""
    config = evaluation_env_config(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=10,
        max_steps=3,
    )
    context = evaluation_context(
        expected_horizon=2,
        config=config,
    )
    state, observation, action_mask, _reset_info = reset(
        config,
        jax.random.PRNGKey(0),
    )
    state, observation, _reward, _done, action_mask, _step_info = step(
        config,
        state,
        action_mask,
        neutral_action(),
        jax.random.PRNGKey(1),
    )
    state = state._replace(team_deathmatch_scores=jnp.asarray((2, 1), dtype=jnp.int32))
    team_a_active = jnp.logical_and(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids == 1,
    )
    team_b_active = jnp.logical_and(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids == 2,
    )
    context_features = observation.context_features
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_ALLY_SCORE].set(
        jnp.where(
            team_a_active,
            2.0,
            jnp.where(team_b_active, 1.0, 0.0),
        )
    )
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_ENEMY_SCORE].set(
        jnp.where(
            team_a_active,
            1.0,
            jnp.where(team_b_active, 2.0, 0.0),
        )
    )
    observation = observation._replace(context_features=context_features)
    current_frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
    )
    frames = [current_frame]
    transitions: list[EvaluationTransitionV1] = []
    for transition_index in range(context.expected_horizon):
        state, observation, reward, done, action_mask, info = step(
            config,
            state,
            action_mask,
            neutral_action(),
            jax.random.PRNGKey(transition_index + 2),
        )
        transition, current_frame = capture_evaluation_transition_unit_v1(
            context,
            current_frame,
            state,
            observation,
            action_mask,
            info.transition_facts,
            reward,
            done,
        )
        transitions.append(transition)
        frames.append(current_frame)
    return CapturedEvaluationTrajectory(
        context=context,
        frames=tuple(frames),
        transitions=tuple(transitions),
    )


__all__ = [
    "CapturedEvaluationTrajectory",
    "captured_evaluation_trajectory",
    "captured_resumed_team_deathmatch_horizon_trajectory",
    "captured_team_deathmatch_threshold_trajectory",
    "content_identity",
    "evaluation_context",
    "evaluation_env_config",
    "evaluation_episode_identity",
    "evaluation_seed_protocol",
    "mage_target_none_ultimate_action",
    "neutral_action",
    "policy_assignments",
    "valid_shared_availability",
]
