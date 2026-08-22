"""Core type contracts for the JAX-native simulator spine."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# Prevents association between “Team A” and padding.
NO_TEAM_ID = 0
TEAM_A_ID = 1
TEAM_B_ID = 2
NUM_MOVE_ACTIONS = 9
NUM_ULTIMATE_ACTIONS = 2
NUM_TEAMS = 2
MAX_AGENTS_PER_TEAM = 5
MAX_AGENT_SLOTS = NUM_TEAMS * MAX_AGENTS_PER_TEAM
NUM_TARGET_ACTIONS = MAX_AGENT_SLOTS + 1
ENVIRONMENT_DIMENSIONS = 2
MAX_OBSTACLE_SLOTS = 16
OBSTACLE_FEATURES = 8
OBSTACLE_TYPE_NONE = 0
OBSTACLE_TYPE_PILLAR = 1
OBSTACLE_TYPE_WALL = 2
OBSTACLE_FEATURE_TYPE = 0
OBSTACLE_FEATURE_X = 1
OBSTACLE_FEATURE_Y = 2
OBSTACLE_FEATURE_RADIUS = 3
OBSTACLE_FEATURE_WIDTH = 4
OBSTACLE_FEATURE_HEIGHT = 5
OBSTACLE_FEATURE_THETA = 6
OBSTACLE_FEATURE_ACTIVE = 7
MOVE_STAY = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_EAST = 3
MOVE_WEST = 4
MOVE_NORTHEAST = 5
MOVE_NORTHWEST = 6
MOVE_SOUTHEAST = 7
MOVE_SOUTHWEST = 8
CLASS_NEUTRAL = 0
NEUTRAL_CLASS_ID = CLASS_NEUTRAL
MAGE_CLASS_ID = 1
WARRIOR_CLASS_ID = 2
HUNTER_CLASS_ID = 3
ROGUE_CLASS_ID = 4
PRIEST_CLASS_ID = 5
NUM_CLASSES = 6
SELF_FEATURES = 58
UNIT_FEATURES = 58
MAX_OBJECTIVE_SLOTS = 8
OBJECTIVE_FEATURES = 12
CONTEXT_FEATURES = 19

# Fixed numeric task modes keep the traced simulator free of strings and registries.
NUM_TASKS = 3
TASK_MODE_NEUTRAL = 0
TASK_MODE_TDM = 1
TASK_MODE_KOTH = 2
TASK_MODE_CTF = 3

# Task outcomes are shared semantics across every battleground mode.
TASK_MODE_OUTCOME_ONGOING = 0
TASK_MODE_OUTCOME_TEAM_A_WIN = 1
TASK_MODE_OUTCOME_TEAM_B_WIN = 2
TASK_MODE_OUTCOME_DRAW = 3

# Canonical sparse task rewards.
REWARD_FOR_WINNING = 1
REWARD_FOR_LOSING = -1
REWARD_FOR_DRAWING = 0

# Context exposes raw simulator and task facts. Canonical learner-facing
# normalization belongs to the later versioned observation-preprocessing layer.
# Mode-specific score and threshold fields are temporary schema reservations;
# their final semantics remain owned by the corresponding battleground modes.
CONTEXT_FEATURE_CURRENT_TIMESTEP = 0
CONTEXT_FEATURE_EPISODE_HORIZON = 1
CONTEXT_FEATURE_MAP_WIDTH = 2
CONTEXT_FEATURE_MAP_HEIGHT = 3
CONTEXT_FEATURE_ALLY_TEAM_SIZE = 4
CONTEXT_FEATURE_ENEMY_TEAM_SIZE = 5
CONTEXT_FEATURE_IS_TDM = 6
CONTEXT_FEATURE_IS_KOTH = 7
CONTEXT_FEATURE_IS_CTF = 8
CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT = 9
CONTEXT_FEATURE_TDM_ALLY_SCORE = 10
CONTEXT_FEATURE_TDM_ENEMY_SCORE = 11
CONTEXT_FEATURE_KOTH_ALLY_SCORE = 12
CONTEXT_FEATURE_KOTH_ENEMY_SCORE = 13
CONTEXT_FEATURE_CTF_ALLY_CAPTURE_COUNT = 14
CONTEXT_FEATURE_CTF_ENEMY_CAPTURE_COUNT = 15
CONTEXT_FEATURE_TDM_SCORE_THRESHOLD = 16
CONTEXT_FEATURE_KOTH_SCORE_THRESHOLD = 17
CONTEXT_FEATURE_CTF_CAPTURE_THRESHOLD = 18

# Self rows and unit-candidate rows use one shared agent-feature schema.
# self_features exists only for convenient actor conditioning; ally/enemy unit
# rows are relation-indexed candidate uses of the same AGENT_FEATURE_* contract.
# Keep SELF_FEATURES == UNIT_FEATURES unless a future schema decision explicitly
# splits these families.
AGENT_FEATURE_X = 0
AGENT_FEATURE_Y = 1
AGENT_FEATURE_RADIUS = 2
AGENT_FEATURE_TEAM_ID = 3
AGENT_FEATURE_ACTIVE = 4
AGENT_FEATURE_ALIVE = 5
AGENT_FEATURE_CLASS_ID = 6
AGENT_FEATURE_BASE_MOVEMENT_SPEED = 7
AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED = 8
AGENT_FEATURE_OBSERVATION_RADIUS = 9
AGENT_FEATURE_BASIC_INTERACTION_RADIUS = 10
AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS = 11
AGENT_FEATURE_CURRENT_HEALTH = 12
AGENT_FEATURE_MAX_HEALTH = 13
AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING = 14

# Effect features use EFFECT_CLASS_ABILITY_TYPE so adjacent columns group by
# tactical meaning while keeping the source explicit for researchers.

# Agent is under these slows. Durations use 0 while inactive; multipliers use
# the multiplicative identity 1.0 while inactive.
AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION = 15
AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION = 16
AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION = 17
AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER = 18
AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER = 19
AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER = 20

# Agent is under these stuns: 0 when not active.
# Stuns do not stack, they run concurrently.
AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION = 21
AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION = 22
AGENT_FEATURE_STUN_ROGUE_POISON_DURATION = 23

# Agent is under this debuff. Duration uses 0 and multiplier uses 1.0 while
# inactive.
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION = 24
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER = 25

# Agent is under this buff. 0 when burst not active.
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION = 26

# Agent is under this buff.
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION = 27
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION = 28

# Dynamic out-of-combat status sits beside the other policy-visible timers.
AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT = 29

# Everything below is no longer part of state.
# Agent is under these aura modifiers.
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER = 30
AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER = 31

# Describe row-local capabilities of the agent. Capability multipliers use
# 0.0 for absence; they are payload descriptors, not active effective values.
AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE = 32
AGENT_FEATURE_CAPABILITY_BASIC_HEALING = 33
AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION = 34

AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION = 35
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION = 36
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION = 37
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER = 38
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER = 39
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER = 40

AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION = 41
AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION = 42
AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION = 43

AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION = 44
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER = 45

AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION = 46
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER = 47

AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION = 48
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION = 49

AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS = 50
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER = 51
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS = 52
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER = 53

AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING = 54
AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE = 55

AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS = 56
AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP = 57

NUM_SLOW_CHANNELS = 3
SLOW_CHANNEL_WARRIOR_CHARGE = 0
SLOW_CHANNEL_HUNTER_BASIC = 1
SLOW_CHANNEL_ROGUE_POISON = 2

NUM_STUN_CHANNELS = 3
STUN_CHANNEL_WARRIOR_CHARGE = 0
STUN_CHANNEL_HUNTER_TRAP = 1
STUN_CHANNEL_ROGUE_POISON = 2


class ResolvedAgentProfile(NamedTuple):
    """Immutable fixed-slot facts resolved before ordinary reset."""

    class_ids: Array
    team_ids: Array
    active_mask: Array
    agent_radii: Array
    base_movement_speeds: Array
    observation_radii: Array
    basic_interaction_radii: Array
    ultimate_interaction_radii: Array
    max_health: Array
    out_of_combat_delay_steps: Array  # (MAX_AGENT_SLOTS,)
    out_of_combat_health_regen_fraction_per_step: Array  # (MAX_AGENT_SLOTS,)


class EnvConfig(NamedTuple):
    """Static episode settings and reset inputs.

    Static map geometry belongs here because Milestone 4 obstacles do not change
    during an episode. ``agent_profile`` is the sole authority for immutable
    fixed-slot roster topology and capabilities resolved before ordinary reset.
    ``ordinary_movement_distance_scale`` converts catalog movement speeds into
    per-step voluntary displacement without scaling forced relocation.
    Team-local spawn pads are the sole ordinary-reset position authority.
    Spawn-shield duration and movement speed remain episode-static rules.
    Team respawn-wave periods are immutable public clocks whose current
    countdowns live in ``EnvState``.
    ``task_mode`` selects one fixed JAX task branch, while the Team Deathmatch
    threshold remains an exact host-validated integer episode rule.
    """

    task_mode: int
    team_deathmatch_score_threshold: int
    max_steps: int
    map_width: float
    map_height: float
    obstacles: Array
    agent_profile: ResolvedAgentProfile
    ordinary_movement_distance_scale: float
    team_spawn_pad_positions: Array  # (NUM_TEAMS, MAX_AGENTS_PER_TEAM, 2)
    spawn_shield_duration_steps: int
    spawn_shield_movement_speed: float
    team_respawn_wave_period_step_count: Array  # (NUM_TEAMS,)


class EnvState(NamedTuple):
    """Dynamic slot-aligned simulator state carried through transitions.

    Episode-static facts live in ``EnvConfig.agent_profile``. Status state keeps
    only source-specific remaining durations; fixed magnitudes are derived by
    ``core.combat.derive_status_magnitudes``. Previous-action fields retain the
    one accepted category per actor from the immediately preceding transition.
    The spawn-shield vector stores remaining protected movement steps without a
    duplicate active flag. Team respawn-wave countdowns store one public clock
    per team without duplicating eligibility or queue state. The scalar validity
    leaf distinguishes reset from a real neutral action.
    Team Deathmatch scores use the fixed order ``[Team A, Team B]`` and remain
    integral so simultaneous deaths can update both totals without attribution.
    """

    team_deathmatch_scores: Array  # (NUM_TEAMS,)
    step_count: Array
    agent_positions: Array
    alive_mask: Array
    current_health: Array
    ultimate_cooldowns: Array
    slow_durations: Array
    stun_durations: Array
    rogue_poison_anti_heal_durations: Array
    mage_burst_damage_amplification_durations: Array
    priest_blessing_of_freedom_slow_floor_durations: Array
    team_respawn_wave_countdowns: Array
    spawn_shield_durations: Array
    steps_until_out_of_combat: Array  # (MAX_AGENT_SLOTS,)
    previous_timestep_move_actions: Array
    previous_timestep_select_target_actions: Array
    previous_timestep_use_ultimate_actions: Array
    has_previous_timestep_joint_action: Array


class Action(NamedTuple):
    """Factored joint action supplied by policies for every agent slot."""

    move: Array
    select_target: Array
    use_ultimate: Array


class ActionMask(NamedTuple):
    """Protocol-admissible choices for the slot-aligned action interface.

    The joint target/ultimate mask is authoritative for exact combat pairs.
    The flat target and ultimate masks are its existential marginals and support
    per-head policy masking without independently defining legality. Dead and
    inactive slots expose only the canonical effect-inert submission; mask
    membership does not by itself imply physical agency or participation.
    """

    move_mask: Array
    select_target_mask: Array
    use_ultimate_mask: Array
    select_target_use_ultimate_joint_mask: Array


class PreviousTimestepActionObservation(NamedTuple):
    """Observer-relative one-hot history for actors visible in the current state.

    Every leaf has axes ``(observer, observed-actor relation row, category)``.
    Ally and enemy relation rows match the corresponding unit-feature tensors.
    Reset and hidden-actor rows are all zero; visible rows from a real
    transition contain one accepted category per action head.
    """

    ally_previous_timestep_move_actions_one_hot: Array
    enemy_previous_timestep_move_actions_one_hot: Array
    ally_previous_timestep_select_target_actions_one_hot: Array
    enemy_previous_timestep_select_target_actions_one_hot: Array
    ally_previous_timestep_use_ultimate_actions_one_hot: Array
    enemy_previous_timestep_use_ultimate_actions_one_hot: Array


class SpawnLifecycleObservation(NamedTuple):
    """Actor-relative public spawn, shield, roster, and respawn-clock truth.

    Team A observers receive Team A before Team B; Team B observers receive
    Team B before Team A. Configured class-to-slot rows remain public through
    occlusion, death, shielding, and respawn. Configured inactive observer rows
    remain canonical zeros across every leaf.
    """

    spawn_pad_positions_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM, 2)
    )
    spawn_shield_actual_durations_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    spawn_shield_configured_duration_by_agent: Array  # (MAX_AGENT_SLOTS)
    spawn_shield_speed_by_agent: Array  # (MAX_AGENT_SLOTS)
    respawn_wave_period_step_count_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS)
    )
    respawn_wave_countdowns_by_agent_by_team: Array  # (MAX_AGENT_SLOTS, NUM_TEAMS)
    active_mask_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    alive_mask_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    class_ids_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )


class Observation(NamedTuple):
    """Structured per-slot observations emitted by reset and step.

    Unit features and visibility use stable observer-relative rows. Previous
    actions are a separate categorical family and do not extend the shared
    agent-feature columns.
    """

    self_features: Array
    ally_unit_features: Array
    enemy_unit_features: Array
    map_obstacle_features: Array
    objective_features: Array
    context_features: Array  # Meta/Config features.
    ally_visibility_mask: Array
    enemy_visibility_mask: Array
    previous_timestep_actions: PreviousTimestepActionObservation
    spawn_lifecycle: SpawnLifecycleObservation


class Reward(NamedTuple):
    """Slot-aligned scalar rewards emitted by the core simulator."""

    rewards: Array  # (MAX_AGENT_SLOTS,)


class DoneFlags(NamedTuple):
    """Episode termination and truncation signals from the core simulator."""

    terminated: Array
    truncated: Array

    @property
    def done(self) -> Array:
        """Whether rollout control should stop for this episode."""
        return jnp.logical_or(self.terminated, self.truncated)


class ActionAcceptanceFacts(NamedTuple):
    """Submitted intent, accepted behavior, and per-actor rejection provenance."""

    submitted_joint_action: Action
    accepted_joint_action: Action
    submitted_action_tuple_is_out_of_domain_by_actor: Array
    in_domain_move_action_is_rejected_by_actor: Array
    in_domain_combat_action_pair_is_rejected_by_actor: Array


class CombatTransitionFacts(NamedTuple):
    """Source-aligned accepted effects and authoritative recipient totals."""

    basic_effect_is_activated_by_source: Array
    ultimate_effect_is_activated_by_source: Array
    combat_effect_has_recipient_by_source: Array
    combat_effect_recipient_global_slot_by_source: Array
    raw_damage_output_by_source: Array
    source_modified_damage_output_by_source: Array
    recipient_damage_modifier_by_source: Array
    total_effective_damage_by_recipient: Array
    raw_healing_output_by_source: Array
    source_modified_healing_output_by_source: Array
    recipient_healing_modifier_by_source: Array
    total_effective_healing_by_recipient: Array
    health_after_combat_resolution_by_recipient: Array
    slow_is_applied_by_source_and_channel: Array
    stun_is_applied_by_source_and_channel: Array
    rogue_poison_anti_heal_is_applied_by_source: Array
    mage_burst_damage_amplification_is_applied_by_source: Array
    priest_blessing_of_freedom_is_applied_by_source: Array


class DeathTransitionFacts(NamedTuple):
    """New successor deaths and their positive effective-damage contributors.

    Recipient truth is aligned by global slot. Contributor truth and attributed
    gross damage are aligned by source slot; attribution is neither killer
    selection nor realized-health-loss apportionment.
    """

    is_newly_dead_by_recipient: Array
    contributed_to_new_death_by_source: Array
    attributed_death_damage_by_source: Array


class SpawnShieldTransitionFacts(NamedTuple):
    """Authoritative spawn-shield activity and ordinary-expiry facts."""

    was_active_at_transition_start_by_agent: Array  # (MAX_AGENT_SLOTS,)
    expired_at_transition_end_by_agent: Array  # (MAX_AGENT_SLOTS,)


class RespawnTransitionFacts(NamedTuple):
    """Authoritative due-wave and realized-respawn facts for one transition."""

    respawn_wave_occurred_this_transition_by_team: Array  # (NUM_TEAMS,)
    was_respawned_this_transition_by_agent: Array  # (MAX_AGENT_SLOTS,)


class RegenerationTransitionFacts(NamedTuple):
    """Authoritative combat-countdown and regeneration facts for one transition."""

    combat_countdown_was_reset_by_agent: Array  # (MAX_AGENT_SLOTS,)
    actual_health_regenerated_this_step_by_agent: Array  # (MAX_AGENT_SLOTS,)


class PhysicalTransitionFacts(NamedTuple):
    """Realized per-slot displacement authored at each movement phase."""

    charge_phase_displacement_by_agent: Array  # (MAX_AGENT_SLOTS, 2)
    ordinary_movement_phase_displacement_by_agent: Array  # (MAX_AGENT_SLOTS, 2)


class AuraTransitionFacts(NamedTuple):
    """Transition-start emitter-to-beneficiary aura coverage relations."""

    # (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS)
    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary: Array
    # (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS)
    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary: Array


class StatusLifecycleTransitionFacts(NamedTuple):
    """Independent recipient-aligned causes across nine status channels.

    Columns are Warrior Charge slow, Hunter Basic slow, Rogue Poison slow,
    Warrior Charge stun, Hunter Trap stun, Rogue Poison stun, Rogue Poison
    anti-heal, Mage Burst damage amplification, and Priest Blessing of Freedom
    movement floor, in that order.
    """

    aged_to_zero_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    refreshed_or_extended_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    broken_by_damage_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    cleared_by_new_death_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)


class TeamDeathmatchTransitionFacts(NamedTuple):
    """Task outcome authored by one Team Deathmatch transition.

    The scalar uses the shared task-outcome encoding: ongoing ``0``, Team A
    win ``1``, Team B win ``2``, or draw ``3``.
    """

    outcome: Array


class TransitionFacts(NamedTuple):
    """Fixed-shape authoritative facts for one reset or environment transition."""

    has_transition: Array
    transition_start_step_count: Array
    action_acceptance_facts: ActionAcceptanceFacts
    combat_transition_facts: CombatTransitionFacts
    death_facts: DeathTransitionFacts
    spawn_shield_facts: SpawnShieldTransitionFacts
    respawn_facts: RespawnTransitionFacts
    regeneration_facts: RegenerationTransitionFacts
    physical_facts: PhysicalTransitionFacts
    aura_facts: AuraTransitionFacts
    status_lifecycle_facts: StatusLifecycleTransitionFacts
    team_deathmatch_facts: TeamDeathmatchTransitionFacts


class Info(NamedTuple):
    """Privileged fixed-shape host diagnostics returned by reset and step.

    Transition facts are global simulator truth and are not decentralized
    policy observations.
    """

    transition_facts: TransitionFacts
