"""Define Core array axes, category IDs and immutable simulator records.

EnvConfig holds fixed episode rules; EnvState carries current numerical state.
Action, Observation and ActionMask form the decision interface. Reward and
DoneFlags report task results; Info groups privileged transition facts used by
runner, replay and metric consumers. Nested records document their exact fields.

Shapes describe one ten-slot game unless stated otherwise. Public native
batching adds a leading environment axis. These NamedTuples are JAX-compatible
data containers, not validators, policy adapters or lifecycle managers.
Feature-column and category constants remain the shared schema authority."""

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
MAX_OBSTACLE_SLOTS = 32
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
AGENT_FEATURE_IS_ENEMY = 3
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
    """Hold resolved roster membership and class capabilities for one episode.

    Every field is a JAX array (10,), with Team A in slots 0..4 and Team B in
    5..9. Each team uses an active prefix followed by unused slots. Public
    native preparation adds a leading B axis to every leaf. Capabilities come
    from the class catalog; this record is not another place to change rules.

    Notes
    -----
    All fields are required. The NamedTuple is immutable and construction only
    packages values. Host configuration validation checks shapes, dtypes,
    neutral padding and exact catalog agreement. Inactive slots have class/team
    ID zero and zero capabilities. Death does not change this episode profile.
    """

    class_ids: Array
    """Configured class IDs for the ten fixed slots.

    Int32 (10,): 0 for unused slots; 1 Mage, 2 Warrior, 3 Hunter, 4 Rogue or 5
    Priest for configured agents.
    """
    team_ids: Array
    """Configured team IDs for the ten fixed slots.

    Int32 (10,): 1 for active Team A slots, 2 for active Team B slots, and 0
    for unused slots. This is runner configuration, not actor identity input.
    """
    active_mask: Array
    """Configured membership for the ten fixed slots.

    Bool (10,): configured membership, unchanged by death or respawn. True
    entries form a prefix within each five-slot team block.
    """
    agent_radii: Array
    """Catalog body radii for the ten fixed slots.

    Float32 (10,): catalog body radii in world units; unused slots are zero.
    """
    base_movement_speeds: Array
    """Catalog movement speeds before scaling and status effects.

    Float32 (10,): catalog movement speeds before the ordinary movement scale
    and current status effects; unused slots are zero.
    """
    observation_radii: Array
    """Catalog observation radii for the ten fixed slots.

    Float32 (10,): catalog observation radii in world units; unused slots are
    zero. Current visibility also depends on simulator rules.
    """
    basic_interaction_radii: Array
    """Catalog Basic interaction radii for the ten fixed slots.

    Float32 (10,): catalog Basic interaction radii in world units; unused slots
    are zero. A radius alone does not establish target legality.
    """
    ultimate_interaction_radii: Array
    """Catalog Ultimate interaction radii for the ten fixed slots.

    Float32 (10,): catalog Ultimate interaction radii in world units; unused
    slots are zero. A radius alone does not establish target legality.
    """
    max_health: Array
    """Catalog maximum health for the ten fixed slots.

    Float32 (10,): catalog maximum health; unused slots are zero.
    """
    out_of_combat_delay_steps: Array  # (MAX_AGENT_SLOTS,)
    """Catalog recovery delay for the ten fixed slots.

    Int32 (10,): catalog recovery delay in steps, in 0..16,777,216; unused
    slots are zero. The current remaining delay belongs to EnvState.
    """
    out_of_combat_health_regen_fraction_per_step: Array  # (MAX_AGENT_SLOTS,)
    """Catalog health recovery fraction per eligible step.

    Float32 (10,): catalog fraction of maximum health recovered per eligible
    step, in [0.0, 1.0]; unused slots are zero.
    """


class EnvConfig(NamedTuple):
    """Hold episode rules, map geometry and the ordinary reset inputs.

    agent_profile owns fixed roster membership and catalog capabilities.
    team_spawn_pad_positions owns the ordered ordinary-reset positions.
    ordinary_movement_distance_scale converts class speeds into voluntary
    distance per step; it does not scale forced relocation. Shield rules and
    team wave periods remain fixed while their current counters live in EnvState.

    Notes
    -----
    All fields are required and have no constructor defaults. Task factories
    own defaults; host Core validation owns shapes, types, bounds and geometry.
    Constructing this immutable tuple performs no validation. Host checks raise
    TypeError for wrong types/dtypes and ValueError for invalid values or shapes.

    Field docs describe one game. Public native preparation adds a leading B
    axis to every leaf and converts scalar settings to arrays. Fixed within an
    episode does not mean JIT-static: values remain dynamic JAX inputs. task_mode
    selects a fixed branch structure using a dynamic numeric mode. Authored
    starts use a separately validated state while retaining these episode rules.
    """

    task_mode: int
    """The task selected for one episode.

    Python int for one host config: 0 neutral or 1 Team Deathmatch. Modes 2
    (KOTH) and 3 (CTF) are reserved and rejected by current validation.
    """
    team_deathmatch_score_threshold: int
    """The score threshold used to end a Team Deathmatch episode.

    Python int: 0 in neutral mode; 1..16,777,212 in Team Deathmatch. The bound
    keeps the threshold and one-step score overshoot exact in float32.
    """
    max_steps: int
    """The episode step limit.

    Python int in 1..16,777,216: episode step limit. Reaching it sets
    truncated, including when termination also occurs on that step.
    """
    map_width: float
    """The map width in world units.

    Python float: finite positive world width, at most the float32 maximum.
    Core validates body and spawn-pad clearance within the map.
    """
    map_height: float
    """The map height in world units.

    Python float: finite positive world height, at most the float32 maximum.
    Core validates body and spawn-pad clearance within the map.
    """
    obstacles: Array
    """Fixed obstacle geometry for one episode.

    Float32 JAX array (32, 8), with finite rows ordered as [type, x, y, radius,
    width, height, theta, active]. Type 1 is a pillar; type 2 is a wall. Active
    is 0 or 1. Unused rows are all zero; active geometry must satisfy Core
    rules. Theta uses radians.
    """
    agent_profile: ResolvedAgentProfile
    """The resolved roster and capabilities for ten fixed slots.

    ResolvedAgentProfile of ten fixed slots. Core validates the roster, zero
    padding and class-catalog capabilities; Team Deathmatch needs at least one
    configured agent per team.
    """
    ordinary_movement_distance_scale: float
    """The scale applied to voluntary movement distance.

    Python float: finite and in (0.0, 1.0] after float32 conversion. It scales
    voluntary displacement, not forced relocation. Product validation requires
    the canonical value 1.0.
    """
    team_spawn_pad_positions: Array  # (NUM_TEAMS, MAX_AGENTS_PER_TEAM, 2)
    """Ordered world positions for each team's five spawn pads.

    Float32 JAX array (2, 5, 2) of finite world [x, y] positions. Team order is
    [Team A, Team B]; pad order follows each team roster. Core checks all pads,
    including unused ones, for map/obstacle clearance and overlapping fallback
    bodies. Ordinary reset uses this order.
    """
    spawn_shield_duration_steps: int
    """The number of protected movement steps after a wave respawn.

    Python int in 0..2,147,483,647: protected movement steps after a wave
    respawn. Zero disables the shield. Ordinary reset starts without an active
    shield.
    """
    spawn_shield_movement_speed: float
    """The configured movement speed while a spawn shield is active.

    Python float: shielded movement speed, finite and positive after float32
    conversion. This remains a valid positive setting even when
    spawn_shield_duration_steps is zero.
    """
    team_respawn_wave_period_step_count: Array  # (NUM_TEAMS,)
    """The fixed respawn-wave period for each team.

    Int32 JAX array (2,) in [Team A, Team B] order, with values
    1..2,147,483,647 steps. Each current countdown starts at period minus one;
    the countdown itself belongs to EnvState.
    """


class EnvState(NamedTuple):
    """Carry the current ten-slot simulator state between transitions.

    Episode rules and fixed capabilities live in EnvConfig. This record keeps
    current positions, health, counters and compact accepted-action history.
    Status channels store remaining durations; combat derives their strengths.
    Previous actions belong to the transition that just ended, and the scalar
    validity flag distinguishes reset from a real accepted neutral action.

    Notes
    -----
    All fields are required JAX arrays. The NamedTuple is immutable and does not
    validate values. Scalar-game field shapes are given below; public native
    batching adds a leading B axis. Carry the latest state together with its
    matching ActionMask. Host state validation requires a matching valid config.
    Dead agents retain cooldowns and accepted history, but health, shields and
    transient statuses are zero. This global state is privileged runner data;
    it is not an actor's decentralized observation.
    """

    team_deathmatch_scores: Array  # (NUM_TEAMS,)
    """Current integer team scores.

    Int32 (2,) in [Team A, Team B] order. Neutral mode uses zero. Valid TDM
    snapshots stay at or below threshold plus four; both teams may score in one
    transition.
    """
    step_count: Array
    """The current episode step.

    Nonnegative int32 scalar (). Ordinary reset starts at zero and step increments
    once. Authored starts may retain a nonzero value. This is an episode step, not a
    runner-wide transition counter.
    """
    agent_positions: Array
    """Current body centers in world coordinates.

    Float32 (10, 2) [x, y], in Team A then Team B slot order. Unused slots are zero.
    Configured dead bodies retain positions but do not participate in physical
    collision.
    """
    alive_mask: Array
    """Current alive status for the ten fixed slots.

    Bool (10,). True requires configured membership. Death clears this flag; a due
    respawn restores it without changing the configured roster.
    """
    current_health: Array
    """Current health after the latest completed transition.

    Float32 (10,). Living configured slots have health in (0, max_health]; dead and
    unused slots have zero. Respawn restores class maximum health.
    """
    ultimate_cooldowns: Array
    """Remaining steps before Ultimate is ready.

    Int32 (10,) in [0, class cooldown]. Zero means ready. Accepted use starts the
    full catalog cooldown; other counters age once. Death retains cooldowns; unused
    slots are zero.
    """
    slow_durations: Array
    """Remaining slow steps in three source channels.

    Int32 (10, 3), in Warrior Charge, Hunter Basic, Rogue Poison order. Counts are
    nonnegative and bounded by their catalog durations. Zero means no effect in that
    channel; dead and unused rows are zero.
    """
    stun_durations: Array
    """Remaining stun steps in three source channels.

    Int32 (10, 3), in Warrior Charge, Hunter Trap, Rogue Poison order. Counts are
    nonnegative and catalog-bounded. Any positive channel controls the current
    decision; dead and unused rows are zero.
    """
    rogue_poison_anti_heal_durations: Array
    """Remaining Rogue Poison healing-reduction steps.

    Nonnegative catalog-bounded int32 (10,). Zero means ordinary healing. Dead and
    unused slots are zero; combat derives the multiplier from this counter.
    """
    mage_burst_damage_amplification_durations: Array
    """Remaining Mage Burst damage-amplification steps.

    Nonnegative catalog-bounded int32 (10,). Only configured Mages may have a
    positive counter. Dead and unused slots are zero. Fresh application first
    affects the next decision.
    """
    priest_blessing_of_freedom_slow_floor_durations: Array
    """Remaining Priest movement-floor steps.

    Nonnegative catalog-bounded int32 (10,). Positive values enable the configured
    minimum movement fraction without clearing slow channels. Dead and unused slots
    are zero.
    """
    team_respawn_wave_countdowns: Array
    """Current public respawn-wave clocks for both teams.

    Int32 (2,) in [Team A, Team B] order, each in [0, period - 1]. Zero means a wave
    is due in the next transition. Clocks advance even when a due wave has no
    eligible actors.
    """
    spawn_shield_durations: Array
    """Remaining protected movement steps after respawn.

    Int32 (10,) in [0, configured shield duration]. Zero means unshielded. The
    current counter controls movement and visibility; a value of one rejoins body
    contact at the final movement endpoint. Dead and unused slots are zero.
    """
    steps_until_out_of_combat: Array  # (MAX_AGENT_SLOTS,)
    """Remaining steps before out-of-combat recovery is eligible.

    Int32 (10,) in [0, resolved class delay]. Participation resets the delay;
    otherwise it ages toward zero. Recovery reads the current counter before ageing.
    Dead and unused slots are zero.
    """
    previous_timestep_move_actions: Array
    """Accepted movement from the transition that just ended.

    Int32 (10,) categories 0..8 in global actor order. Reset and unused rows are
    zero. A dead actor may retain its last accepted category.
    """
    previous_timestep_select_target_actions: Array
    """Accepted targets from the transition that just ended.

    Int32 (10,) categories 0..10, relative to each acting slot's own team. Reset and
    unused rows are zero. Observation construction remaps target categories for
    opposing observers.
    """
    previous_timestep_use_ultimate_actions: Array
    """Accepted Ultimate choices from the transition that just ended.

    Int32 (10,) categories zero or one in global actor order. Reset and unused rows
    are zero; death does not erase accepted history.
    """
    has_previous_timestep_joint_action: Array
    """Whether the state follows a real simulator transition.

    Bool scalar (). False makes all stored history heads canonical zero at reset.
    True distinguishes a real accepted neutral action from no history.
    """


class Action(NamedTuple):
    """Hold one submitted movement, target and Ultimate choice for every slot.

    All three fields are int32 JAX arrays (10,) for one game, or (B, 10) through
    a native environment. Team A occupies slots 0..4 and Team B occupies 5..9,
    including unused slots. ActorAction is the separate one-actor scalar type.

    Notes
    -----
    Choose this action from the current permitted Observation and ActionMask,
    then pass it to step. Targets are relative to each acting agent's team;
    movement directions retain world coordinates. The exact target/Ultimate
    mask owns pair legality. Valid per-head ranges alone are insufficient,
    and admitted actions need not cause a physical effect.

    All fields are required. The immutable tuple only packages values; it does
    not validate, sample, cast or carry policy memory. Core records submitted
    and accepted actions separately when categories are rejected.
    """

    move: Array
    """The submitted movement category for each fixed actor slot.

    Int32 (10,), or (B, 10): 0 Stay, 1 North, 2 South, 3 East, 4 West, 5
    Northeast, 6 Northwest, 7 Southeast, 8 Southwest. Directions use world
    coordinates; the current move_mask still applies.
    """
    select_target: Array
    """The submitted target category for each fixed actor slot.

    Int32 (10,), or (B, 10): 0 Target None; 1..5 select own-team roster rows
    0..4, including self; 6..10 select opposing rows 0..4. Check the chosen
    target/Ultimate pair against the current joint mask.
    """
    use_ultimate: Array
    """The submitted Ultimate category for each fixed actor slot.

    Int32 (10,), or (B, 10): 0 does not request Ultimate; 1 requests it. This
    is a categorical int32 head, not a bool array. Its effect depends on the
    accepted target/Ultimate pair and the actor class.
    """


class ActionMask(NamedTuple):
    """Describe the choices admitted for the current ten-slot decision.

    The joint target/Ultimate mask is the authority for exact combat pairs.
    The flat masks mark categories with at least one compatible partner;
    two True marginal entries need not form a True joint entry. A legal move
    can still be blocked by physics. Dead and unused slots admit only Stay
    and (Target None, no Ultimate), without granting them physical agency.

    Notes
    -----
    Every leaf is bool. Field shapes describe one game; native batching adds
    a leading B axis and selecting one actor removes the ten-slot axis.
    Reset returns the initial mask. Step returns the successor mask with its
    matching Observation; use that pair for the next decision.
    All fields are required. This immutable tuple does not validate agreement,
    sample actions or grant access to other actors' private inputs.
    """

    move_mask: Array
    """Movement categories admitted at the current decision.

    Bool (10, 9): admitted movement categories for the current decision. True
    does not promise nonzero displacement or collision-free movement.
    """
    select_target_mask: Array
    """Targets admitted with at least one Ultimate choice.

    Bool (10, 11): True when at least one Ultimate choice is allowed with that
    target. This marginal alone cannot validate a combat pair.
    """
    use_ultimate_mask: Array
    """Ultimate choices admitted with at least one target.

    Bool (10, 2): True when at least one target is allowed with that Ultimate
    choice. This marginal alone cannot validate a combat pair.
    """
    select_target_use_ultimate_joint_mask: Array
    """Exact target and Ultimate pairs admitted at this decision.

    Bool (10, 11, 2): the authority for exact target/Ultimate pairs, indexed by
    [actor, target category, Ultimate category].
    """


class PreviousTimestepActionObservation(NamedTuple):
    """Hold visible accepted action history in each observer's roster order.

    Float32 leaves contain zero/one categories, with axes [observer, relation
    row, category]. The first two axes have sizes ten and five. Ally/enemy rows
    match unit features. Native batching adds B; one actor omits the first axis.

    Notes
    -----
    History describes the transition that just ended, not the next action.
    Current visibility of the observed actor gates a whole row. Target visibility
    does not hide that actor's accepted target identity. Opposing observers swap
    target relation blocks so they still decode the same physical target.
    Reset and hidden rows are all zero; a real accepted category zero is one-hot.
    All fields are required. The immutable tuple performs no history update,
    validation or filtering by itself.
    """

    ally_previous_timestep_move_actions_one_hot: Array
    """Visible allies' accepted movement from the last transition.

    Float32 (10, 5, 9): visible allies' accepted movement categories, in stable
    own-team roster order; zero when history is absent or hidden.
    """
    enemy_previous_timestep_move_actions_one_hot: Array
    """Visible enemies' accepted movement from the last transition.

    Float32 (10, 5, 9): visible enemies' accepted movement categories, in
    stable opposing-team roster order; zero when history is absent or hidden.
    """
    ally_previous_timestep_select_target_actions_one_hot: Array
    """Visible allies' accepted targets from the last transition.

    Float32 (10, 5, 11): visible allies' accepted target categories, expressed
    in the observer's relation rows; zero for absent/hidden history.
    """
    enemy_previous_timestep_select_target_actions_one_hot: Array
    """Visible enemies' accepted targets from the last transition.

    Float32 (10, 5, 11): visible enemies' accepted target categories, remapped
    to the observer's relation rows; zero for absent/hidden history.
    """
    ally_previous_timestep_use_ultimate_actions_one_hot: Array
    """Visible allies' accepted Ultimate use from the last transition.

    Float32 (10, 5, 2): visible allies' accepted Ultimate categories 0 or 1;
    zero when history is absent or hidden.
    """
    enemy_previous_timestep_use_ultimate_actions_one_hot: Array
    """Visible enemies' accepted Ultimate use from the last transition.

    Float32 (10, 5, 2): visible enemies' accepted Ultimate categories 0 or 1;
    zero when history is absent or hidden.
    """


class SpawnLifecycleObservation(NamedTuple):
    """Hold current public spawn, shield, roster and respawn-clock facts.

    Each field starts with ten observers. Team axes use [own team, opponent],
    and roster axes use five stable rows. Team A sees Team A first; Team B sees
    Team B first. Positions retain world coordinates without reflection.
    Native batching adds B and one actor omits the observer axis.

    Notes
    -----
    Configured living and dead observers retain this public information through
    occlusion, shielding and respawn. Unused observer rows are zero, but active
    observers still see all five ordered pads, including pads for unused slots.
    Config owns fixed rules/classes/membership; state owns current alive flags,
    shield durations and wave countdowns. Values describe the current decision.
    All fields are required; the immutable tuple does not validate or update them.
    """

    spawn_pad_positions_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM, 2)
    )
    """Public spawn-pad positions in each observer's team order.

    Float32 (10, 2, 5, 2): public world [x, y] spawn pads, ordered by observer,
    relative team, roster row and coordinate. Unused observers are zero;
    configured observers see all ordered pads.
    """
    spawn_shield_actual_durations_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    """Current shield steps remaining in each observer's team order.

    Int32 (10, 2, 5): current shield steps remaining, in 0..configured
    duration. Zero means no remaining shield; unused observers are zero.
    """
    spawn_shield_configured_duration_by_agent: Array  # (MAX_AGENT_SLOTS)
    """The public configured shield duration for each observer.

    Int32 (10,): public configured shield duration, in 0..2,147,483,647 steps.
    Zero disables shielding; unused observers also receive zero.
    """
    spawn_shield_speed_by_agent: Array  # (MAX_AGENT_SLOTS)
    """The public configured shielded movement speed for each observer.

    Float32 (10,): public configured shielded movement speed, positive for
    configured observers and zero for unused observers.
    """
    respawn_wave_period_step_count_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS)
    )
    """Public wave periods in each observer's team order.

    Int32 (10, 2): public own-team and opposing-team wave periods, in
    1..2,147,483,647 steps for configured observers; otherwise zero.
    """
    respawn_wave_countdowns_by_agent_by_team: Array  # (MAX_AGENT_SLOTS, NUM_TEAMS)
    """Current wave countdowns in each observer's team order.

    Int32 (10, 2): current own-team and opposing-team clocks, in 0..period
    minus one. Zero marks a wave due in the next transition; it does not grant
    a dead actor an action before respawn. Unused observers receive zero.
    """
    active_mask_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    """Public roster membership in each observer's team order.

    Bool (10, 2, 5): configured roster membership. Death does not clear
    membership. Unused observers receive False throughout.
    """
    alive_mask_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    """Current public alive flags in each observer's team order.

    Bool (10, 2, 5): current public alive flags in roster order. Unused roster
    slots and unused observer rows are False.
    """
    class_ids_by_agent_by_team: (
        Array  # (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM)
    )
    """Public configured classes in each observer's team order.

    Int32 (10, 2, 5): public configured class IDs, 1..5 for active roster rows
    and 0 for unused slots. Death does not change them. Unused observers
    receive zero throughout.
    """


class Observation(NamedTuple):
    """Hold structured decision inputs for all ten fixed observer slots.

    Field shapes describe one game. Native batching adds B to every leaf,
    including nested families; selecting one actor removes the observer axis.
    Unit rows use stable own-team and opposing-team roster order. self_ally_index
    identifies self in the five own-team rows, not a simulator team or global ID.

    Notes
    -----
    Reset returns the initial input. Step returns the successor input and its
    matching ActionMask. Current features and masks describe the same decision;
    the separate previous-action family describes the transition that just ended.
    Coordinates retain the world frame, without reflection. Features are raw
    float32, not normalized; this module's feature constants name their columns.
    Hidden unit rows are zero. Public geometry remains present even for unused
    observers, so a complete unused Observation need not be all zero.

    Each actor may use only its permitted row. Holding this whole runner payload
    does not grant permission to share private observations or memory. All fields
    are required; the immutable tuple does not build, validate, filter or flatten
    data. env.observation_space(agent) describes one actor's field structure.
    """

    self_features: Array
    """Each observer's own current agent features.

    Float32 (10, 58): each observer's own current agent features.
    AGENT_FEATURE_* constants name the columns; status and capability values
    keep their documented world units, health units or step counts.
    """
    ally_unit_features: Array
    """Visible own-team agent features in stable roster order.

    Float32 (10, 5, 58): own-team rows using the self-feature schema and stable
    roster order, including self. Nonvisible rows are zero.
    """
    enemy_unit_features: Array
    """Visible opposing-team agent features in stable roster order.

    Float32 (10, 5, 58): opposing-team rows using the self-feature schema and
    stable roster order. Nonvisible rows are zero.
    """
    map_obstacle_features: Array
    """Full public obstacle geometry for every observer.

    Float32 (10, 32, 8): full public obstacle rows copied from
    EnvConfig.obstacles for every observer, including unused observers. Columns
    keep the configuration schema and world coordinates.
    """
    objective_features: Array
    """Reserved objective features for every observer.

    Float32 (10, 8, 12): reserved objective rows. The current neutral and Team
    Deathmatch implementations emit zeros.
    """
    context_features: Array  # Meta/Config features.
    """Raw current context and episode rules for every observer.

    Float32 (10, 19): raw current step, episode rules, relative team sizes,
    task flags, scores and thresholds. CONTEXT_FEATURE_* constants name the
    columns. Reserved task/objective values and unused observer rows are zero.
    """
    ally_visibility_mask: Array
    """Current visibility of each stable own-team row.

    Bool (10, 5): current visibility of each stable own-team row. Use it to
    distinguish a hidden zeroed unit row from observed data.
    """
    enemy_visibility_mask: Array
    """Current visibility of each stable opposing-team row.

    Bool (10, 5): current visibility of each stable opposing-team row. Use it
    to distinguish a hidden zeroed unit row from observed data.
    """
    previous_timestep_actions: PreviousTimestepActionObservation
    """Visible accepted action history from the last transition.

    PreviousTimestepActionObservation: float32 one-hot accepted action history,
    filtered by current observed-actor visibility. Leaves have shape (10, 5,
    category count); reset history is all zero.
    """
    spawn_lifecycle: SpawnLifecycleObservation
    """Current public spawn rules, clocks and roster facts.

    SpawnLifecycleObservation: current public spawn pads, shield rules, shield
    durations, wave clocks, roster classes, membership and alive flags. Each
    leaf starts with the same ten-observer axis.
    """
    self_ally_index: Array  # int32 (MAX_AGENT_SLOTS,), scalar for one actor.
    """The observer's own row within its five own-team rows.

    Int32 (10,) in 0..4: self's row in own-team observations and target
    categories. It is scalar for one actor and zero for unused slots; it is not
    a global slot or simulator team ID.
    """


class Reward(NamedTuple):
    """Hold the reward assigned to every configured or unused actor slot.

    Notes
    -----
    The required rewards field is float32 (10,) for one game; native batching
    adds B. Team Deathmatch emits +1/-1 to configured winning/losing slots,
    including dead agents. Ongoing, draw and unused slots receive zero.
    This immutable tuple packages values without validation or terminal latching.
    """

    rewards: Array  # (MAX_AGENT_SLOTS,)
    """Reward for each fixed actor slot.

    Float32 (10,), or (B, 10) through native batching. Configured dead agents share
    the team result; unused slots receive zero.
    """


class DoneFlags(NamedTuple):
    """Hold separate task-completion and time-limit flags for one episode.

    Bool leaves have scalar shape () for one game or (B,) for native batching.
    There is no actor axis; an individual death or respawn is separate from
    episode completion. Core step computes these flags from the successor state.

    Notes
    -----
    Team Deathmatch terminates when either score reaches its threshold. Reaching
    the horizon truncates. Both may be True together. Neutral mode uses only
    truncation. Flags alone do not identify a winner or choose a learning
    algorithm's value-bootstrap rule. The done property combines them with JAX.
    All fields are required; the immutable tuple does not validate or auto-reset.
    """

    terminated: Array
    """Whether the task has ended in the successor state.

    Bool scalar (), or (B,): task completion in the successor state. Team
    Deathmatch sets this when either score reaches the threshold; it can be
    True together with truncated.
    """
    truncated: Array
    """Whether the successor state has reached the episode step limit.

    Bool scalar (), or (B,): the successor step count has reached the episode
    limit. This can coincide with task termination and does not by itself
    identify a draw or a winner.
    """

    @property
    def done(self) -> Array:
        """Return whether rollout control should stop for this episode.

        Returns
        -------
        jax.Array
            Bool scalar () for one game or bool (B,) for native batching: the
            elementwise OR of terminated and truncated. Both original flags remain
            available for separate learning decisions.

        Notes
        -----
        This JAX property does not read device values on the host, decide a winner
        or reset the environment. It is safe to use inside JAX transforms.
        """
        return jnp.logical_or(self.terminated, self.truncated)


class ActionAcceptanceFacts(NamedTuple):
    """Record submitted actions, accepted actions and each actor's rejection cause.

    The two Actions retain global actor-slot order. Rejection vectors are bool
    (10,). Out-of-domain rejection is separate from in-domain mask rejection;
    target/Ultimate rejection applies to the pair. Facts describe one transition
    and do not grant policy information rights. All fields are required and
    construction only packages values; native batching adds B to every leaf.
    """

    submitted_joint_action: Action
    """The joint action received before category acceptance.

    Action with three int32 (10,) heads. Values retain the submission, including IDs
    rejected as out of range.
    """
    accepted_joint_action: Action
    """The action Core actually uses for this transition.

    Action with three int32 (10,) heads. Rejected entries are replaced with their
    canonical neutral submission before effect resolution.
    """
    submitted_action_tuple_is_out_of_domain_by_actor: Array
    """Whether any submitted head is outside its category range.

    Bool (10,). True rejects that actor's entire tuple, before mask lookup; this is
    separate from an in-domain masked choice.
    """
    in_domain_move_action_is_rejected_by_actor: Array
    """Whether an in-domain tuple submitted a masked movement.

    Bool (10,). False for out-of-domain tuples, which use the separate domain-
    rejection flag.
    """
    in_domain_combat_action_pair_is_rejected_by_actor: Array
    """Whether an in-domain tuple submitted a masked combat pair.

    Bool (10,). Target and Ultimate are checked together; rejection cannot turn an
    Ultimate request into a Basic action.
    """


class CombatTransitionFacts(NamedTuple):
    """Record accepted source effects and the recipient totals used by combat.

    Source and recipient axes both use global slots, not actor-relative target
    IDs. Float32 magnitudes describe health units or dimensionless modifiers;
    bool arrays record accepted applications. Source totals are gross effects,
    before recipient health clipping, rather than apportioned health loss.
    All fields are required. This immutable record only packages one transition;
    native batching adds B. Neutral reset facts are padding, not observed events.
    """

    basic_effect_is_activated_by_source: Array
    """Whether each source applies an accepted targeted Basic.

    Bool (10,) in global source order. Target None or accepted Ultimate use makes
    this False.
    """
    ultimate_effect_is_activated_by_source: Array
    """Whether each source uses an accepted Ultimate.

    Bool (10,), including no-target Mage Burst. Activation does not imply a routed
    health recipient.
    """
    combat_effect_has_recipient_by_source: Array
    """Whether each accepted target names a global recipient.

    Bool (10,). Target None is False, including a valid no-target Ultimate.
    """
    combat_effect_recipient_global_slot_by_source: Array
    """The global recipient selected by each source.

    Int32 (10,), in 0..9 or -1 for Target None. IDs are global slots, not actor-
    relative target categories.
    """
    raw_damage_output_by_source: Array
    """Accepted catalog damage before any source or recipient modifier.

    Float32 (10,) nonnegative health units. Sources without a damage payload are
    zero.
    """
    source_modified_damage_output_by_source: Array
    """Damage after source Burst and Mage aura modifiers.

    Float32 (10,) nonnegative health units, before recipient mitigation or health
    clipping.
    """
    recipient_damage_modifier_by_source: Array
    """The chosen recipient's damage factor for each damage source.

    Float32 (10,) dimensionless Warrior mitigation factors. Sources without positive
    raw routed damage use zero, not an identity factor.
    """
    total_effective_damage_by_recipient: Array
    """Gross incoming damage after source and recipient modifiers.

    Float32 (10,) nonnegative health units in global recipient order. Totals precede
    net healing and health clipping; they are not realized health loss.
    """
    raw_healing_output_by_source: Array
    """Accepted catalog healing before recipient modification.

    Float32 (10,) nonnegative health units. Sources without a healing payload are
    zero.
    """
    source_modified_healing_output_by_source: Array
    """Healing after source modification and before recipient modification.

    Float32 (10,) nonnegative health units. Current mechanics have no healing
    amplifier, so this equals raw healing.
    """
    recipient_healing_modifier_by_source: Array
    """The chosen recipient's healing factor for each healing source.

    Float32 (10,) dimensionless current Poison factors. Sources without positive raw
    routed healing use zero.
    """
    total_effective_healing_by_recipient: Array
    """Gross incoming healing after recipient modifiers.

    Float32 (10,) nonnegative health units in global recipient order, before net
    damage and maximum-health clipping.
    """
    health_after_combat_resolution_by_recipient: Array
    """Health after simultaneous damage and healing are netted and clipped.

    Float32 (10,) in [0, class maximum]. This is before out-of-combat recovery and
    respawn restoration.
    """
    slow_is_applied_by_source_and_channel: Array
    """Accepted slow applications by source and mechanic channel.

    Bool (10, 3), with Warrior Charge, Hunter Basic and Rogue Poison columns. These
    record applications, even if successor death clears the duration.
    """
    stun_is_applied_by_source_and_channel: Array
    """Accepted stun applications by source and mechanic channel.

    Bool (10, 3), with Warrior Charge, Hunter Trap and Rogue Poison columns. The
    chosen global recipient is stored separately.
    """
    rogue_poison_anti_heal_is_applied_by_source: Array
    """Accepted Rogue Poison anti-heal applications.

    Bool (10,) in global source order, routed to the source's accepted recipient.
    """
    mage_burst_damage_amplification_is_applied_by_source: Array
    """Accepted Mage Burst applications to the source itself.

    Bool (10,) in global source order. This self-buff has no routed recipient
    target.
    """
    priest_blessing_of_freedom_is_applied_by_source: Array
    """Accepted Priest Basic movement-floor applications.

    Bool (10,) in global source order, routed to the source's accepted recipient.
    """


class DeathTransitionFacts(NamedTuple):
    """Record new deaths and sources that contributed positive effective damage.

    Recipient facts use global recipient slots; contribution and damage facts
    use global source slots. Damage is gross post-source/post-recipient damage,
    not killer selection or an allocation of realized health loss. Simultaneous
    healing and health clipping do not redistribute credit. All fields are
    required arrays (10,) for one transition; native batching adds B.
    The immutable record does not perform attribution itself.
    """

    is_newly_dead_by_recipient: Array
    """Whether a configured start-alive recipient died this transition.

    Bool (10,). Existing corpses do not count again. Death is resolved after health
    effects and recovery, before respawn.
    """
    contributed_to_new_death_by_source: Array
    """Whether a source dealt positive effective damage to a new death.

    Bool (10,). Several sources may contribute to the same recipient; no single
    killer is chosen.
    """
    attributed_death_damage_by_source: Array
    """Gross effective damage attributed to a newly dead recipient.

    Float32 (10,) in health units by source. Zero for no contribution. This can
    exceed realized health loss and is not divided among contributors.
    """


class SpawnShieldTransitionFacts(NamedTuple):
    """Record shield activity at transition start and ordinary expiry at its end.

    Both required fields are bool (10,) in global slot order. Expiry means a
    start counter of one reached its end while the successor actor is alive;
    death clearing and a newly created respawn shield are different events.
    Native batching adds B. The immutable tuple does not update counters.
    """

    was_active_at_transition_start_by_agent: Array  # (MAX_AGENT_SLOTS,)
    """Whether a positive shield counter governed this transition.

    Bool (10,) from the transition-start state, before movement and ageing.
    """
    expired_at_transition_end_by_agent: Array  # (MAX_AGENT_SLOTS,)
    """Whether a one-step shield expired while its actor stayed alive.

    Bool (10,). This marks ordinary expiry, not death clearing or a newly created
    respawn shield.
    """


class RespawnTransitionFacts(NamedTuple):
    """Record due team waves and the actors that respawned in this transition.

    A due wave is recorded even when it has no eligible dead members. Only
    configured actors already dead at transition start may respawn; newly
    killed actors wait for a later wave. All fields are required; the immutable
    tuple only packages facts. Native batching adds B to both leaves.
    """

    respawn_wave_occurred_this_transition_by_team: Array  # (NUM_TEAMS,)
    """Whether each team's start countdown was zero.

    Bool (2,) in [Team A, Team B] order. True includes an empty due wave.
    """
    was_respawned_this_transition_by_agent: Array  # (MAX_AGENT_SLOTS,)
    """Whether each configured start-dead actor respawned at the end.

    Bool (10,) in global slot order. Newly killed actors wait for a later wave.
    """


class RegenerationTransitionFacts(NamedTuple):
    """Record combat-countdown resets and actual out-of-combat health recovery.

    Fields are global-slot arrays (10,) for one transition. Actual recovery
    follows eligibility, current Poison and maximum-health clipping; respawn
    health restoration is excluded. All fields are required. Native batching
    adds B; constructing this immutable tuple does not change health or timers.
    """

    combat_countdown_was_reset_by_agent: Array  # (MAX_AGENT_SLOTS,)
    """Whether accepted participation triggered a combat-delay reset.

    Bool (10,) by global slot. The event remains recorded even when new death clears
    the stored successor countdown.
    """
    actual_health_regenerated_this_step_by_agent: Array  # (MAX_AGENT_SLOTS,)
    """Actual health gained through out-of-combat recovery.

    Float32 (10,) nonnegative health units after eligibility, Poison and maximum-
    health clipping. No recovery is zero; respawn restoration is excluded.
    """


class PhysicalTransitionFacts(NamedTuple):
    """Record realized displacement separately for Charge and ordinary movement.

    Both required fields are float32 (10, 2) world [dx, dy] values. Charge
    displacement ends at the scene where ordinary movement begins. The later
    respawn position override is not included in either movement phase.
    Native batching adds B. This immutable record does not project positions.
    """

    charge_phase_displacement_by_agent: Array  # (MAX_AGENT_SLOTS, 2)
    """Realized world displacement during simultaneous Charge placement.

    Float32 (10, 2), measured from transition-start positions to the post-Charge
    scene, including contact correction.
    """
    ordinary_movement_phase_displacement_by_agent: Array  # (MAX_AGENT_SLOTS, 2)
    """Realized world displacement during ordinary movement.

    Float32 (10, 2), measured from the post-Charge scene to the movement endpoint.
    The later respawn override is excluded.
    """


class AuraTransitionFacts(NamedTuple):
    """Record transition-start aura coverage from each emitter to each beneficiary.

    Both required bool (10, 10) fields use global emitter rows and beneficiary
    columns. Eligible configured living unshielded allies include self, with
    inclusive radius checks. Coverage records each emitter before stacking caps.
    Native batching adds B. The immutable record does not grant visibility or
    policy access to these global relations.
    """

    # (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS)
    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary: Array
    """Which Mage emitters cover which beneficiaries at transition start.

    Bool (10, 10), with global emitter rows and beneficiary columns. This records
    coverage before the outgoing-damage stacking cap.
    """
    # (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS)
    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary: Array
    """Which Warrior emitters cover beneficiaries at transition start.

    Bool (10, 10), with global emitter rows and beneficiary columns. This records
    coverage before the incoming-damage stacking floor.
    """


class StatusLifecycleTransitionFacts(NamedTuple):
    """Record independent recipient causes across nine ordered status channels.

    Columns are Warrior Charge slow, Hunter Basic slow, Rogue Poison slow,
    Warrior Charge stun, Hunter Trap stun, Rogue Poison stun, Rogue Poison
    anti-heal, Mage Burst damage amplification, and Priest Blessing of Freedom
    movement floor, in that order. Each required field is bool (10, 9), with
    global recipient rows; native batching adds B.

    Notes
    -----
    Causes need not be mutually exclusive. Age old durations before damage
    break and new applications; then clear surviving durations on new death.
    Expiry followed by application or Trap break followed by application is
    distinct from refreshing a still-positive duration. This immutable record
    packages causes without deriving them from the final snapshot afterward.
    """

    aged_to_zero_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    """Whether ordinary ageing reduced an old duration of one to zero.

    Bool (10, 9), before damage break, fresh applications and death clearing. A
    later reapplication does not erase this cause.
    """
    refreshed_or_extended_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    """Whether an application restored or extended a still-positive status.

    Bool (10, 9). The old status must not have expired or broken; its new pre-death
    duration must reach at least the old duration.
    """
    broken_by_damage_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    """Whether positive raw damage cleared an aged existing Trap.

    Bool (10, 9). Only Hunter Trap can be True in the current catalog. Fresh Trap
    application occurs later and is not broken retroactively.
    """
    cleared_by_new_death_by_recipient_and_status_channel: Array  # (MAX_AGENT_SLOTS, 9)
    """Whether new death cleared a duration left after applications.

    Bool (10, 9). Only positive pre-death successor durations count; earlier expiry
    or break is a separate cause.
    """


class TeamDeathmatchTransitionFacts(NamedTuple):
    """Record the Team Deathmatch result of one completed simulator transition.

    The required outcome is an int32 scalar: 0 ongoing, 1 Team A wins,
    2 Team B wins, 3 draw. Native batching adds B. Neutral initialization uses
    zero because no transition occurred; check TransitionFacts.has_transition.
    This immutable record does not compute scores, rewards or episode flags.
    """

    outcome: Array
    """The categorical Team Deathmatch result.

    Int32 scalar (): 0 ongoing, 1 Team A wins, 2 Team B wins, 3 draw. A false
    has_transition makes the zero value neutral padding.
    """


class TransitionFacts(NamedTuple):
    """Group authoritative facts for one transition or an absent-transition row.

    has_transition distinguishes a real simulator step from initialization or
    padding. The start-step field names the decision that acted; nested facts
    record the corresponding phases, even though the returned state is its
    successor. Global source/recipient truth is privileged runner data.

    Notes
    -----
    All fields are required. This immutable NamedTuple only packages values.
    Native batching adds B to every leaf. When has_transition is False, the
    start step and recipient IDs use -1 and remaining facts use zero/False.
    Consumers must not interpret these neutral values as a measured transition.
    """

    has_transition: Array
    """Whether these facts describe a real simulator step.

    Bool scalar (). False marks initialization or padding, whose nested zero values
    are not measured events.
    """
    transition_start_step_count: Array
    """The episode step from which the recorded action was taken.

    Int32 scalar (), nonnegative for a real transition and -1 when has_transition is
    False. The returned state carries the successor step.
    """
    action_acceptance_facts: ActionAcceptanceFacts
    """Submitted and accepted actions with per-actor rejection causes."""
    combat_transition_facts: CombatTransitionFacts
    """Source applications and the recipient totals used by combat."""
    death_facts: DeathTransitionFacts
    """New recipient deaths and their positive source contributions."""
    spawn_shield_facts: SpawnShieldTransitionFacts
    """Shield activity at the start and ordinary expiry at the end."""
    respawn_facts: RespawnTransitionFacts
    """Due team waves and realized end-of-transition respawns."""
    regeneration_facts: RegenerationTransitionFacts
    """Combat-delay reset causes and actual recovery health gained."""
    physical_facts: PhysicalTransitionFacts
    """Separate realized Charge and ordinary world displacements."""
    aura_facts: AuraTransitionFacts
    """Transition-start global emitter-to-beneficiary coverage."""
    status_lifecycle_facts: StatusLifecycleTransitionFacts
    """Independent recipient status-ageing, break, refresh and death causes."""
    team_deathmatch_facts: TeamDeathmatchTransitionFacts
    """The Team Deathmatch result after scoring this transition."""


class Info(NamedTuple):
    """Hold privileged fixed-shape diagnostics returned by Core reset and step.

    The required transition_facts field contains global simulator truth.
    It is not a decentralized Observation and does not grant policy access.
    Initialization returns an absent-transition record; step returns facts for
    the transition just completed. Native batching adds B to nested leaves.
    The immutable tuple does not log, serialize, validate or advance anything.
    """

    transition_facts: TransitionFacts
    """Global diagnostic facts for a transition or an absent-transition row.

    Check has_transition before interpreting nested event values. These facts are
    privileged simulator data, separate from actor observations.
    """
