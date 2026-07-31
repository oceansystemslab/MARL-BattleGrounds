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
SELF_FEATURES = 55
UNIT_FEATURES = 55
MAX_OBJECTIVE_SLOTS = 8
OBJECTIVE_FEATURES = 12
CONTEXT_FEATURES = 19

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
CONTEXT_FEATURE_IS_TEAM_DEATHMATCH = 6
CONTEXT_FEATURE_IS_KING_OF_THE_HILL = 7
CONTEXT_FEATURE_IS_CAPTURE_THE_FLAG = 8
CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT = 9
CONTEXT_FEATURE_TEAM_DEATHMATCH_ALLY_SCORE = 10
CONTEXT_FEATURE_TEAM_DEATHMATCH_ENEMY_SCORE = 11
CONTEXT_FEATURE_KING_OF_THE_HILL_ALLY_SCORE = 12
CONTEXT_FEATURE_KING_OF_THE_HILL_ENEMY_SCORE = 13
CONTEXT_FEATURE_CAPTURE_THE_FLAG_ALLY_CAPTURE_COUNT = 14
CONTEXT_FEATURE_CAPTURE_THE_FLAG_ENEMY_CAPTURE_COUNT = 15
CONTEXT_FEATURE_TEAM_DEATHMATCH_SCORE_THRESHOLD = 16
CONTEXT_FEATURE_KING_OF_THE_HILL_SCORE_THRESHOLD = 17
CONTEXT_FEATURE_CAPTURE_THE_FLAG_CAPTURE_THRESHOLD = 18

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

# Everything below is no longer part of state.
# Agent is under these aura modifiers.
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER = 29
AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER = 30

# Describe row-local capabilities of the agent. Capability multipliers use
# 0.0 for absence; they are payload descriptors, not active effective values.
AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE = 31
AGENT_FEATURE_CAPABILITY_BASIC_HEALING = 32
AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION = 33

AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION = 34
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION = 35
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION = 36
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER = 37
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER = 38
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER = 39

AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION = 40
AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION = 41
AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION = 42

AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION = 43
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER = 44

AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION = 45
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER = 46

AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION = 47
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION = 48

AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS = 49
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER = 50
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS = 51
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER = 52

AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING = 53
AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE = 54

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


class EnvConfig(NamedTuple):
    """Static episode settings and reset inputs.

    Static map geometry belongs here because Milestone 4 obstacles do not change
    during an episode. ``agent_profile`` is the sole authority for immutable
    fixed-slot roster topology and capabilities resolved before ordinary reset.
    ``ordinary_movement_distance_scale`` converts catalog movement speeds into
    per-decision voluntary displacement without scaling forced relocation.
    """

    max_steps: int
    map_width: float
    map_height: float
    obstacles: Array
    agent_profile: ResolvedAgentProfile
    initial_agent_positions: Array
    ordinary_movement_distance_scale: float


class EnvState(NamedTuple):
    """Dynamic slot-aligned simulator state carried through transitions.

    Episode-static facts live in ``EnvConfig.agent_profile``. Status state keeps
    only source-specific remaining durations; fixed magnitudes are derived by
    ``core.combat.derive_status_magnitudes``. Previous-action fields retain the
    one accepted category per actor from the immediately preceding transition.
    The scalar validity leaf distinguishes reset from a real neutral action.
    """

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
    context_features: Array
    ally_visibility_mask: Array
    enemy_visibility_mask: Array
    previous_timestep_actions: PreviousTimestepActionObservation


class Reward(NamedTuple):
    """Slot-aligned scalar rewards emitted by the core simulator."""

    rewards: Array


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


class TransitionFacts(NamedTuple):
    """Fixed-shape authoritative facts for one reset or environment transition."""

    has_transition: Array
    choosing_step_count: Array
    action_acceptance_facts: ActionAcceptanceFacts
    combat_transition_facts: CombatTransitionFacts
    death_facts: DeathTransitionFacts


class Info(NamedTuple):
    """Privileged fixed-shape host diagnostics returned by reset and step.

    Transition facts are global simulator truth and are not decentralized
    policy observations.
    """

    transition_facts: TransitionFacts
