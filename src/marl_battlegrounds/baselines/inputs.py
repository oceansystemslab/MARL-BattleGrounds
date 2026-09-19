"""Encode permitted actor inputs and separate physical training state.

The two encoders provide fixed float32 feature layouts for baseline networks.
Actor features retain each recipient's own information rights. Training-state
features are privileged and must never enter an actor's network or memory.
The ordered layouts below own the versioned offsets; Core still owns field and
category meanings. This module needs only the ordinary JAX environment install.
Changing feature order, category meaning, masking or numeric representation
requires a new version of the affected schema.

Values keep their existing units. Integer conversion uses ordinary float32
rounding: integers above 16,777,216 are not all represented exactly. Encoding is
therefore not a lossless serialization of the full supported Core integer range.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_CLASSES,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_TYPE_WALL,
    TASK_MODE_CTF,
    TEAM_B_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.policies.input import ActorInput

ACTOR_INPUT_SCHEMA_VERSION = 1
TRAINING_STATE_SCHEMA_VERSION = 1

# Each entry names a source field, its per-record shape and its encoded width.
type _Layout = tuple[tuple[str, tuple[int, ...], int], ...]

_ACTOR_LAYOUT: _Layout = (
    ("observation.self_features", (58,), 63),
    ("observation.ally_unit_features", (5, 58), 315),
    ("observation.enemy_unit_features", (5, 58), 315),
    ("observation.map_obstacle_features", (32, 8), 320),
    ("observation.objective_features", (8, 12), 96),
    ("observation.context_features", (19,), 19),
    ("observation.ally_visibility_mask", (5,), 5),
    ("observation.enemy_visibility_mask", (5,), 5),
    (
        "observation.previous_timestep_actions.ally_previous_timestep_move_actions_one_hot",
        (5, 9),
        45,
    ),
    (
        "observation.previous_timestep_actions.enemy_previous_timestep_move_actions_one_hot",
        (5, 9),
        45,
    ),
    (
        "observation.previous_timestep_actions.ally_previous_timestep_select_target_actions_one_hot",
        (5, 11),
        55,
    ),
    (
        "observation.previous_timestep_actions.enemy_previous_timestep_select_target_actions_one_hot",
        (5, 11),
        55,
    ),
    (
        "observation.previous_timestep_actions.ally_previous_timestep_use_ultimate_actions_one_hot",
        (5, 2),
        10,
    ),
    (
        "observation.previous_timestep_actions.enemy_previous_timestep_use_ultimate_actions_one_hot",
        (5, 2),
        10,
    ),
    ("observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team", (2, 5, 2), 20),
    (
        "observation.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team",
        (2, 5),
        10,
    ),
    ("observation.spawn_lifecycle.spawn_shield_configured_duration_by_agent", (), 1),
    ("observation.spawn_lifecycle.spawn_shield_speed_by_agent", (), 1),
    (
        "observation.spawn_lifecycle.respawn_wave_period_step_count_by_agent_by_team",
        (2,),
        2,
    ),
    ("observation.spawn_lifecycle.respawn_wave_countdowns_by_agent_by_team", (2,), 2),
    ("observation.spawn_lifecycle.active_mask_by_agent_by_team", (2, 5), 10),
    ("observation.spawn_lifecycle.alive_mask_by_agent_by_team", (2, 5), 10),
    ("observation.spawn_lifecycle.class_ids_by_agent_by_team", (2, 5), 60),
    ("observation.self_ally_index", (), 5),
    ("source_bank.unit_features_by_source_and_candidate", (5, 10, 58), 3150),
    ("source_bank.unit_visibility_by_source_and_candidate", (5, 10), 50),
    ("source_bank.objective_features_by_source", (5, 8, 12), 480),
    ("source_availability", (5,), 5),
)

_TRAINING_STATE_LAYOUT: _Layout = (
    ("state.team_deathmatch_scores", (2,), 2),
    ("state.step_count", (), 1),
    ("state.agent_positions", (10, 2), 20),
    ("state.alive_mask", (10,), 10),
    ("state.current_health", (10,), 10),
    ("state.ultimate_cooldowns", (10,), 10),
    ("state.slow_durations", (10, 3), 30),
    ("state.stun_durations", (10, 3), 30),
    ("state.rogue_poison_anti_heal_durations", (10,), 10),
    ("state.mage_burst_damage_amplification_durations", (10,), 10),
    ("state.priest_blessing_of_freedom_slow_floor_durations", (10,), 10),
    ("state.team_respawn_wave_countdowns", (2,), 2),
    ("state.spawn_shield_durations", (10,), 10),
    ("state.steps_until_out_of_combat", (10,), 10),
    ("state.previous_timestep_move_actions", (10,), 90),
    ("state.previous_timestep_select_target_actions", (10,), 110),
    ("state.previous_timestep_use_ultimate_actions", (10,), 20),
    ("state.has_previous_timestep_joint_action", (), 1),
    ("config.task_mode", (), 4),
    ("config.team_deathmatch_score_threshold", (), 1),
    ("config.max_steps", (), 1),
    ("config.map_width", (), 1),
    ("config.map_height", (), 1),
    ("config.obstacles", (32, 8), 320),
    ("config.agent_profile.class_ids", (10,), 60),
    ("config.agent_profile.team_ids", (10,), 30),
    ("config.agent_profile.active_mask", (10,), 10),
    ("config.agent_profile.agent_radii", (10,), 10),
    ("config.agent_profile.base_movement_speeds", (10,), 10),
    ("config.agent_profile.observation_radii", (10,), 10),
    ("config.agent_profile.basic_interaction_radii", (10,), 10),
    ("config.agent_profile.ultimate_interaction_radii", (10,), 10),
    ("config.agent_profile.max_health", (10,), 10),
    ("config.agent_profile.out_of_combat_delay_steps", (10,), 10),
    ("config.agent_profile.out_of_combat_health_regen_fraction_per_step", (10,), 10),
    ("config.ordinary_movement_distance_scale", (), 1),
    ("config.team_spawn_pad_positions", (2, 5, 2), 20),
    ("config.spawn_shield_duration_steps", (), 1),
    ("config.spawn_shield_movement_speed", (), 1),
    ("config.team_respawn_wave_period_step_count", (2,), 2),
)


def _offsets(layout: _Layout) -> Mapping[str, slice]:
    """Return immutable zero-based feature slices in the declared layout order.

    layout entries contain a field name, input suffix shape and output width.
    Slice stops are exclusive. This runs only while importing schema metadata.
    """
    result: dict[str, slice] = {}
    start = 0
    for name, _, width in layout:
        result[name] = slice(start, start + width)
        start += width
    return MappingProxyType(result)


ACTOR_FEATURE_OFFSETS = _offsets(_ACTOR_LAYOUT)
TRAINING_STATE_FEATURE_OFFSETS = _offsets(_TRAINING_STATE_LAYOUT)
ACTOR_FEATURE_SIZE = sum(width for _, _, width in _ACTOR_LAYOUT)
TRAINING_STATE_FEATURE_SIZE = sum(width for _, _, width in _TRAINING_STATE_LAYOUT)


def _read_field(root: object, path: str) -> ArrayLike:
    """Read one explicitly named numerical field from nested input records.

    root is an ActorInput, EnvState or EnvConfig. path uses dotted attribute
    names from a fixed layout; this does not traverse arbitrary PyTree order.
    Return the supplied array or Python configuration scalar without copying it.
    """
    value = root
    for name in path.split("."):
        value = getattr(value, name)
    return cast(ArrayLike, value)


def _checked_array(
    value: ArrayLike,
    leading: tuple[int, ...],
    suffix: tuple[int, ...],
    name: str,
    *,
    broadcast: bool = False,
) -> Array:
    """Check static field axes and optionally broadcast configuration axes.

    value is a numerical field; leading names its recipient/game/time axes and
    suffix its fixed per-record axes. name labels shape errors. With broadcast
    False, all axes must match exactly. True permits right-aligned broadcasting
    of leading axes only, without changing suffix axes. Return a JAX array.
    Wrong shapes raise ValueError in eager use or tracing, with no data fetch.
    Dtypes and numerical values remain caller preconditions.
    """
    array = jnp.asarray(value)
    shape = (*leading, *suffix)
    if array.shape == shape:
        return array
    if broadcast and array.ndim >= len(suffix):
        actual_suffix = array.shape[-len(suffix) :] if suffix else ()
        actual_leading = array.shape[: -len(suffix)] if suffix else array.shape
        if actual_suffix == suffix:
            try:
                if jnp.broadcast_shapes(actual_leading, leading) == leading:
                    return jnp.broadcast_to(array, shape)
            except ValueError:
                pass
    raise ValueError(f"{name} must have shape {shape}; received {array.shape}")


def _categories(value: Array, count: int, present: Array) -> Array:
    """Encode in-range category IDs, clearing absent entries to all zero.

    value and present have the same leading shape; value contains integer IDs
    in [0, count), and present is Boolean. Return float32 (*value.shape, count).
    Category zero stays distinct from absence. Values are not host-validated.
    """
    return jnp.where(
        present[..., None],
        jax.nn.one_hot(value.astype(jnp.int32), count, dtype=jnp.float32),
        0.0,
    )


def _unit_features(values: Array, visible: Array) -> Array:
    """Replace class IDs in current unit rows while preserving missing rows.

    values ends in 58 Core features; visible matches the preceding axes and
    marks permitted, visible rows. Membership further gates all rows. Return
    float32 rows ending in 63 features, with the six class indicators replacing
    the original class column. Death alone does not erase configured self data.
    """
    present = visible & (values[..., AGENT_FEATURE_ACTIVE] > 0)
    categories = _categories(values[..., AGENT_FEATURE_CLASS_ID], NUM_CLASSES, present)
    encoded = jnp.concatenate(
        (
            values[..., :AGENT_FEATURE_CLASS_ID],
            categories,
            values[..., AGENT_FEATURE_CLASS_ID + 1 :],
        ),
        axis=-1,
    )
    return jnp.where(present[..., None], encoded, 0.0)


def _obstacle_features(values: Array) -> Array:
    """Replace obstacle type with three indicators and clear unused rows.

    values ends in (32, 8) Core obstacle entries. Return float32 rows ending in
    (32, 10), with [None, Pillar, Wall] before the seven remaining Core columns.
    Only the existing active flag admits a row; no geometry is recalculated.
    """
    present = values[..., OBSTACLE_FEATURE_ACTIVE] > 0
    types = _categories(
        values[..., OBSTACLE_FEATURE_TYPE], OBSTACLE_TYPE_WALL + 1, present
    )
    return jnp.where(
        present[..., None],
        jnp.concatenate((types, values[..., OBSTACLE_FEATURE_TYPE + 1 :]), axis=-1),
        0.0,
    )


def _pack(
    fields: Mapping[str, Array], layout: _Layout, leading: tuple[int, ...]
) -> Array:
    """Flatten only declared per-record axes and join fields in schema order.

    fields contains already checked and encoded arrays. layout fixes each
    field's output width; leading axes remain unchanged. Return one float32
    feature axis without any host transfer or persistent expanded-input cache.
    """
    return jnp.concatenate(
        [fields[name].reshape((*leading, width)) for name, _, width in layout],
        axis=-1,
    ).astype(jnp.float32)


def encode_actor_inputs(inputs: ActorInput) -> Array:
    """Turn each actor's permitted current input into 5,164 float32 features.

    Parameters
    ----------
    inputs : ActorInput
        The existing permitted actor input, including its redacted source bank.
        All leaves share leading axes L before their documented per-actor axes.
        L may be empty, (B, 5), or (T, B, 5), for example. Observations and source
        permissions must belong to the same decision. Use the existing input
        builders to enforce information rights; this is not a visibility builder.

    Returns
    -------
    Array
        Float32 shape (*L, ACTOR_FEATURE_SIZE), ordered by ACTOR_FEATURE_OFFSETS
        under ACTOR_INPUT_SCHEMA_VERSION. Each family keeps row-major order.
        Class and obstacle IDs become masked indicators. Other values retain
        their units; existing action-history indicators retain reset absence.
        Public roster facts survive death and occlusion. Unused actors can
        retain public obstacle geometry, so their whole vector need not be zero.

    Raises
    ------
    ValueError
        A field's static shape disagrees with the shared leading axes or schema.

    Notes
    -----
    Inputs are unchanged. The function accepts no privileged state, config or
    random key. It works inside jit/vmap/scan and performs no host data checks,
    I/O, normalization or cross-recipient pooling. Callers supply valid Core
    dtypes/category values. Integer-to-float32 conversion rounds some integers
    above 16,777,216. Keep expanded features temporary at network application;
    store compact observations in rollouts. Learners own inactive loss masking.
    """
    leading = inputs.observation.self_features.shape[:-1]
    fields = {
        name: _checked_array(_read_field(inputs, name), leading, shape, name)
        for name, shape, _ in _ACTOR_LAYOUT
    }
    self_name = "observation.self_features"
    active = fields[self_name][..., AGENT_FEATURE_ACTIVE] > 0
    fields[self_name] = _unit_features(fields[self_name], active)
    for relation in ("ally", "enemy"):
        name = f"observation.{relation}_unit_features"
        fields[name] = _unit_features(
            fields[name], fields[f"observation.{relation}_visibility_mask"]
        )
    obstacle_name = "observation.map_obstacle_features"
    fields[obstacle_name] = _obstacle_features(fields[obstacle_name])
    roster = "observation.spawn_lifecycle."
    class_name = roster + "class_ids_by_agent_by_team"
    fields[class_name] = _categories(
        fields[class_name], NUM_CLASSES, fields[roster + "active_mask_by_agent_by_team"]
    )
    index_name = "observation.self_ally_index"
    fields[index_name] = _categories(fields[index_name], MAX_AGENTS_PER_TEAM, active)
    permission = fields["source_availability"]
    visibility_name = "source_bank.unit_visibility_by_source_and_candidate"
    visibility = fields[visibility_name] & permission[..., :, None]
    fields[visibility_name] = visibility
    unit_name = "source_bank.unit_features_by_source_and_candidate"
    fields[unit_name] = _unit_features(fields[unit_name], visibility)
    objective_name = "source_bank.objective_features_by_source"
    fields[objective_name] = jnp.where(
        permission[..., :, None, None], fields[objective_name], 0.0
    )
    return _pack(fields, _ACTOR_LAYOUT, leading)


def encode_training_state(state: EnvState, config: EnvConfig) -> Array:
    """Encode one privileged physical-state view per game for training only.

    Parameters
    ----------
    state : EnvState
        Current Core state, with common leading axes L before each per-game
        field shape. L may be empty, (B,), or (T, B). Accepted-action history
        describes the completed transition; other fields describe this decision.
    config : EnvConfig
        Matching physical rules and resolved profile. Each config field's leading
        axes may broadcast to L, using ordinary right-aligned broadcasting. For
        example, scalar-game config works with (B,) states, and (B,) config works
        with (T, B) states. Keep values paired with their actual game/decision.

    Returns
    -------
    Array
        Float32 (*L, TRAINING_STATE_FEATURE_SIZE), under
        TRAINING_STATE_SCHEMA_VERSION and TRAINING_STATE_FEATURE_OFFSETS. State
        fields precede config fields in declaration order. Task IDs have four
        indicators, including Neutral and reserved KOTH/CTF; profile team IDs
        have three. Class, team and accepted-action indicators for unused slots
        are zero. Dead members keep physical state and history. All ordered
        spawn pads remain present. No seed, map ID or other runner ID is added.

    Raises
    ------
    ValueError
        State axes disagree, a per-game field shape is wrong, or config leading
        axes cannot broadcast to the state axes.

    Notes
    -----
    Use this view only in a separate critic or mixer. It must not influence
    actor-producing System inputs or memory. Inputs are unchanged; the function
    works in jit/vmap/scan without host data checks or I/O. Core-valid dtypes,
    categories and state/config pairing are caller preconditions. Reserved task
    indicators do not enable those tasks in Core. Boolean fields become 0/1;
    physical values retain units. Integers above 16,777,216 may round in float32.
    Store this view once per game, not once per teammate; any critic broadcast
    belongs at network application. No normalization or recurrent reset occurs.
    """
    leading = state.step_count.shape
    fields: dict[str, Array] = {}
    for name, suffix, _ in _TRAINING_STATE_LAYOUT:
        owner, path = name.split(".", maxsplit=1)
        fields[name] = _checked_array(
            _read_field(state if owner == "state" else config, path),
            leading,
            suffix,
            name,
            broadcast=owner == "config",
        )
    active = fields["config.agent_profile.active_mask"]
    for name, suffix, _ in _TRAINING_STATE_LAYOUT:
        if suffix and suffix[0] == MAX_AGENT_SLOTS:
            mask = active.reshape(
                (*leading, MAX_AGENT_SLOTS, *((1,) * (len(suffix) - 1)))
            )
            fields[name] = jnp.where(mask, fields[name], 0)
    history = active & fields["state.has_previous_timestep_joint_action"][..., None]
    for field, count in (
        ("move", NUM_MOVE_ACTIONS),
        ("select_target", NUM_TARGET_ACTIONS),
        ("use_ultimate", NUM_ULTIMATE_ACTIONS),
    ):
        name = f"state.previous_timestep_{field}_actions"
        fields[name] = _categories(fields[name], count, history)
    task_name = "config.task_mode"
    fields[task_name] = _categories(
        fields[task_name], TASK_MODE_CTF + 1, jnp.ones(leading, dtype=jnp.bool_)
    )
    fields["config.obstacles"] = _obstacle_features(fields["config.obstacles"])
    for field, count in (("class_ids", NUM_CLASSES), ("team_ids", TEAM_B_ID + 1)):
        name = f"config.agent_profile.{field}"
        fields[name] = _categories(fields[name], count, active)
    return _pack(fields, _TRAINING_STATE_LAYOUT, leading)
