"""Discover packaged TDM content and prepare exact episode configurations.

Use list_tdm_maps and list_tdm_scenarios for the installed catalog, the config
factories for ordinary matches, and load_tdm_scenario for an exact authored
start. balanced_spawn_configs prepares both complete spawn-bank choices without
changing team identity, roster order or any other game rule.

Map loading and physical validation happen during host setup. Prepared numerical
configurations can then be selected inside compiled JAX loops. This module does
not choose training distributions, draw random seeds or run games. Core owns
physical validity and simulation rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from operator import itemgetter
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    ScenarioContent,
    TDMAssetSource,
    TDMMapInfo,
    TDMScenarioInfo,
    asset_manifest,
    map_geometry,
    scenario_content,
)
from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
    validate_env_config,
    validate_product_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NUM_CLASSES,
    NUM_TEAMS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_TDM,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
    ResolvedAgentProfile,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
from marl_battlegrounds.evaluation.models import canonical_digest_sha256

type AgentClassName = Literal["mage", "warrior", "hunter", "rogue", "priest"]

_CLASS_IDS: dict[AgentClassName, int] = {
    "mage": MAGE_CLASS_ID,
    "warrior": WARRIOR_CLASS_ID,
    "hunter": HUNTER_CLASS_ID,
    "rogue": ROGUE_CLASS_ID,
    "priest": PRIEST_CLASS_ID,
}
_CANONICAL_ROSTER: tuple[AgentClassName, ...] = tuple(_CLASS_IDS)
CANONICAL_TDM_EVALUATION_MAP_IDS = (47, 48, 49, 50, 51)
# The one default Team Deathmatch Red Zone depth, in map units. Fresh map
# configs, default evaluation, new training runs and new DevClient drafts use it;
# 0.0 keeps one point per death. Saved records keep the depth they recorded.
DEFAULT_TDM_RED_ZONE_DEPTH = 5.0

# These describe Core's array layout only; Core still owns value/geometry checks.
_CONFIG_ARRAY_SHAPES = {
    "obstacles": (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    "team_spawn_pad_positions": (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
        ENVIRONMENT_DIMENSIONS,
    ),
    "team_respawn_wave_period_step_count": (NUM_TEAMS,),
}


@dataclass(frozen=True)
class TDMScenario:
    """Hold one packaged scenario's exact start and source details.

    Attributes
    ----------
    info : TDMScenarioInfo
        Catalog entry containing the scenario ID, name, source identities, team
        roster and expected remaining horizon.
    config : EnvConfig
        Exact resolved scalar configuration for the authored start, including
        its recorded Red Zone depth (5.0 for all eight installed scenarios, so
        a death inside the victim's own Red Zone gives 2 points).
    initial_state : EnvState
        Authored scalar state, including its original step count, positions,
        scores, health and history.
    notes : str
        Authored explanation stored with the scenario.

    Notes
    -----
    load_tdm_scenario validates the configuration, initial state and recorded
    digests. This frozen record does not itself run a game. Pass it as
    env.reset(key, scenario=scenario), or use Core's initialize_scenario_state
    to obtain a validated state/observation/mask snapshot. The config and start
    must remain paired.
    """

    info: TDMScenarioInfo
    config: EnvConfig
    initial_state: EnvState
    notes: str


def list_tdm_maps() -> tuple[TDMMapInfo, ...]:
    """Return the installed map catalog in public ID order.

    Returns
    -------
    tuple[TDMMapInfo, ...]
        The 52 catalog entries: curriculum and other training maps, then
        validation and test maps. Each entry includes source and content identity.

    Notes
    -----
    The package manifest is read and validated once per process, then cached.
    This does not create configurations or load mutable DevClient drafts.
    Package read or validation errors propagate to the caller.
    """
    return asset_manifest().maps


def list_tdm_scenarios() -> tuple[TDMScenarioInfo, ...]:
    """Return the eight installed scenario entries in numeric ID order.

    Returns
    -------
    tuple[TDMScenarioInfo, ...]
        Immutable catalog entries with names, source identities, rosters and
        remaining horizons. To obtain the actual start, call load_tdm_scenario.

    Notes
    -----
    The manifest is cached after its first successful load. This discovery call
    does not play scenarios or read the user's authoring store. Package read
    or validation errors propagate.
    """
    return asset_manifest().scenarios


def canonical_tournament_rosters() -> tuple[
    tuple[AgentClassName, ...], tuple[AgentClassName, ...]
]:
    """Return the default ordered five-agent roster for each team.

    Takes no arguments and performs no device work or configuration validation.
    Factories use this same order when the caller does not supply rosters.

    Returns
    -------
    tuple[tuple[AgentClassName, ...], tuple[AgentClassName, ...]]
        Two immutable tuples, for Team A and Team B. Each contains ``mage``,
        ``warrior``, ``hunter``, ``rogue`` and ``priest`` in that order. Order
        assigns the first five team slots and their matching ordered spawn pads.
        The tuples are shared constants and can be reused safely.
    """
    return _CANONICAL_ROSTER, _CANONICAL_ROSTER


def _config_has_batch(env_config: EnvConfig, num_envs: int | None) -> bool:
    """Check every configuration leaf's scalar or native-batch shape.

    env_config must be exactly EnvConfig with a ResolvedAgentProfile. num_envs
    is the expected native size, or None for scalar execution. Return whether the
    active mask has a native axis. Wrong tuple types raise TypeError; inconsistent
    leaf shapes raise ValueError. Only shapes are inspected, so ordinary numerical
    values can remain dynamic inside JAX. This does not validate physical values.
    """
    if type(env_config) is not EnvConfig:
        raise TypeError("env_config must be an EnvConfig")
    if type(env_config.agent_profile) is not ResolvedAgentProfile:
        raise TypeError("env_config.agent_profile must be a ResolvedAgentProfile")
    active_shape = jnp.shape(env_config.agent_profile.active_mask)
    batched = len(active_shape) == 2
    prefix = (num_envs,) if batched and num_envs is not None else ()
    for name, value in zip(env_config._fields, env_config, strict=True):
        if name == "agent_profile":
            fields = zip(
                env_config.agent_profile._fields, env_config.agent_profile, strict=True
            )
            for field_name, field_value in fields:
                if jnp.shape(field_value) != (*prefix, MAX_AGENT_SLOTS):
                    raise ValueError(
                        f"env_config.agent_profile.{field_name} has an inconsistent "
                        "scalar or batch shape"
                    )
        elif jnp.shape(cast(int | float | Array, value)) != (
            *prefix,
            *_CONFIG_ARRAY_SHAPES.get(name, ()),
        ):
            raise ValueError(
                f"env_config.{name} has an inconsistent scalar or batch shape"
            )
    return batched


def _swap_spawn_banks(env_config: EnvConfig) -> EnvConfig:
    """Exchange the two complete ordered spawn banks in a scalar or batched config.

    env_config uses the validated (..., 2, 5, 2) pad layout. Return an EnvConfig
    with only that field replaced. All ten pads move between teams, including
    inactive slots and later respawn locations. Applying this twice restores the
    original bank order; it is not an idempotent reset.
    """
    return env_config._replace(
        team_spawn_pad_positions=env_config.team_spawn_pad_positions[..., ::-1, :, :]
    )


def _validate_config_choices(
    env_config: EnvConfig, *, batched: bool, both_spawn_choices: bool
) -> None:
    """Validate each distinct concrete source and each requested spawn-bank choice.

    Parameters
    ----------
    env_config : EnvConfig
        A scalar configuration or a batch whose leaf shapes were already checked.
    batched : bool
        Whether each leaf starts with the native batch axis.
    both_spawn_choices : bool
        If True, validate both the supplied and complete exchanged banks. If
        False, validate only the supplied values.

    Returns
    -------
    None
        All requested concrete configurations pass Core's generic validator.

    Raises
    ------
    TypeError, ValueError
        Array dtype conversion would change a dtype, or Core rejects a requested
        configuration. An exchanged-bank failure identifies its source row.

    Notes
    -----
    Tracer inputs are skipped: compiled callers must select from a pool validated
    during setup. Concrete inputs are copied to the host once, and identical rows
    are checked once. Validation prefers CPU vectors for Core's host checks; a
    CUDA-only installation uses its available device. No new physical validator
    or per-step host callback is introduced.
    """
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(env_config)):
        return
    host_config = jax.tree.map(np.asarray, jax.device_get(env_config))
    try:
        validation_device = cast(Any, jax.local_devices(backend="cpu")[0])
    except RuntimeError:
        # JAX_PLATFORMS=cuda can disable the CPU backend entirely.
        validation_device = cast(Any, jax.local_devices()[0])

    def validation_value(value: object) -> object:
        """Prepare one concrete leaf for Core's existing host validator.

        Scalar leaves become Python values. Array leaves move to the selected
        validation device without changing dtype; an unsupported conversion raises
        TypeError. The enclosing setup call owns the device and validated shape.
        """
        host_value = np.asarray(value)
        if host_value.ndim == 0:
            return host_value.item()
        array = jax.device_put(host_value, validation_device)
        if array.dtype != host_value.dtype:
            raise TypeError(
                "env_config array dtype must be supported without conversion"
            )
        return array

    checked: set[tuple[tuple[str, tuple[int, ...], bytes], ...]] = set()
    count = env_config.agent_profile.active_mask.shape[0] if batched else 1
    for index in range(count):
        source = (
            jax.tree.map(itemgetter(index), host_config) if batched else host_config
        )
        identity = tuple(
            (value.dtype.str, value.shape, value.tobytes())
            for value in jax.tree.leaves(source)
        )
        if identity in checked:
            continue
        checked.add(identity)
        # Core requires Python scalars and JAX vectors. CPU vectors let its host
        # checks read values without sending each batch row back from the GPU.
        # A caller may capture a concrete config in a compiled reset closure.
        # Its validation stays in setup rather than tracing Core's host checks.
        with jax.ensure_compile_time_eval(), jax.default_device(validation_device):
            source = jax.tree.map(validation_value, source)
            validate_env_config(source)
            if both_spawn_choices:
                try:
                    validate_env_config(_swap_spawn_banks(source))
                except (TypeError, ValueError) as error:
                    raise type(error)(
                        f"swapped spawn locations in source {index}: {error}"
                    ) from error


def _config_arrays(
    env_config: EnvConfig, *, num_envs: int | None, batched: bool
) -> EnvConfig:
    """Return strongly typed JAX leaves with the requested native axis.

    env_config already has checked shapes. num_envs is None for scalar execution
    or the target positive batch size; batched says whether that axis is already
    present. Convert leaves without changing values. Add an axis only for a scalar
    source entering native execution. The returned configuration does not change
    spawn order or mutate the input.
    """

    def array(value: object) -> Array:
        """Convert one leaf to a JAX array while making its inferred dtype explicit."""
        result = jnp.asarray(value)
        return jnp.asarray(result, dtype=result.dtype)

    arrays = jax.tree.map(array, env_config)
    if num_envs is None or batched:
        return arrays

    def broadcast(value: Array) -> Array:
        """Add the requested native axis without changing the source leaf's values."""
        return jnp.broadcast_to(value, (num_envs, *value.shape))

    return jax.tree.map(broadcast, arrays)


def prepare_exact_env_config(
    env_config: EnvConfig, *, num_envs: int | None
) -> EnvConfig:
    """Validate and prepare exact configuration values for reset.

    Parameters
    ----------
    env_config : EnvConfig
        One resolved ``EnvConfig``, or a native batch with the same
        leading size on every leaf. A single game's spawn pads have shape
        ``(2, 5, 2)``. Custom values accepted by Core's generic validator,
        including movement scale, remain supported. The input is unchanged.
    num_envs : int | None
        Required keyword. ``None`` selects scalar execution; a positive
        integer selects that batch size, including odd sizes. A single config
        broadcasts to the batch. An existing batch must match this size.
        Keep this argument fixed while building a compiled program.

    Returns
    -------
    EnvConfig
        An ``EnvConfig`` whose leaves are JAX arrays with explicit dtypes and,
        for native execution, a leading ``num_envs`` axis. Every value and spawn
        bank stays in its supplied order. Only this exact configuration is
        validated; an unused exchanged bank choice need not be valid.

    Raises
    ------
    TypeError
        A config type or dtype fails the setup or Core contract.
    ValueError
        The batch size, a leaf shape or a Core value is invalid.

    Notes
    -----
    Concrete inputs use Core's host validator once per distinct source. Compiled
    callers must supply previously validated dynamic configs; captured constants
    can be checked while building the program. No host callback runs inside the
    compiled numerical path. This helper changes no input arrays in place.
    """
    if num_envs is not None and (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, Integral)
        or num_envs <= 0
    ):
        raise ValueError("num_envs must be a positive integer or None")
    batched = _config_has_batch(env_config, num_envs)
    _validate_config_choices(env_config, batched=batched, both_spawn_choices=False)
    return _config_arrays(env_config, num_envs=num_envs, batched=batched)


def balanced_spawn_configs(env_config: EnvConfig, *, num_envs: int) -> EnvConfig:
    """Prepare an even batch with source and exchanged spawn locations.

    Parameters
    ----------
    env_config : EnvConfig
        One source ``EnvConfig`` or a full batch of selected sources.
        Each game's spawn pads have shape ``(2, 5, 2)``. Native source leaves
        must all start with ``num_envs``. Sources and their arrays are unchanged.
    num_envs : int
        Required positive even integer. There is no implicit batch size.
        Keep this argument fixed while building a compiled program.

    Returns
    -------
    EnvConfig
        An ``EnvConfig`` with JAX array leaves and one leading ``num_envs`` axis.
        The first half of lanes keep their source's complete ordered spawn banks;
        the second half exchange those two banks. All ten pad rows participate,
        including inactive slots. Rosters, rules, obstacles and other values stay
        unchanged. The helper uses the supplied sources; it does not select maps.

    Raises
    ------
    TypeError
        A config type or dtype fails the setup or Core contract.
    ValueError
        The batch size or a leaf shape is invalid, or Core rejects
        either the source order or its exchanged bank order.

    Notes
    -----
    Concrete setup validates both choices for each distinct source through Core.
    Inside jit, select only from a source pool whose two choices were validated
    on the host. Captured constants can be checked while building the program;
    the compiled path has no host callbacks. This is source preparation, not an
    idempotent reset: passing prepared output back exchanges its second half
    again. Reset consumes the prepared values directly. Use exact preparation
    when only one valid bank choice is wanted.
    """
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, Integral)
        or num_envs <= 0
        or num_envs % 2
    ):
        raise ValueError("num_envs must be a positive even integer")
    batched = _config_has_batch(env_config, num_envs)
    _validate_config_choices(env_config, batched=batched, both_spawn_choices=True)
    arrays = _config_arrays(env_config, num_envs=num_envs, batched=batched)
    swapped = _swap_spawn_banks(arrays)
    return arrays._replace(
        team_spawn_pad_positions=jnp.where(
            (jnp.arange(num_envs) >= num_envs // 2)[:, None, None, None],
            swapped.team_spawn_pad_positions,
            arrays.team_spawn_pad_positions,
        )
    )


def spawn_locations_for_source(
    env_config: EnvConfig, source_config: EnvConfig
) -> tuple[Array, Array]:
    """Check whether resolved configurations use a declared source's spawn banks.

    Parameters
    ----------
    env_config : EnvConfig
        A resolved scalar ``EnvConfig`` or a native batch. Each game's
        pads have shape ``(2, 5, 2)``. All leaves must use the same batch layout.
    source_config : EnvConfig
        The declared source, with unchanged rules, roster and
        geometry. It may be scalar for a scalar or native resolved config,
        or a full batch matching the resolved batch. Its supplied bank order
        defines choice 0, even when that source was itself exchanged earlier.

    Returns
    -------
    tuple[Array, Array]
        ``(matches_source, spawn_choice)`` as Boolean and int32 JAX arrays. Both
        have shape ``()`` for a scalar config or ``(B,)`` for a native batch.
        A match requires every non-pad leaf to be equal and all ten pads to use
        the complete source order or complete exchanged order. Choices are 0 for
        source order, 1 for exchanged order, and -1 for no match or identical
        source banks. ``matches_source`` distinguishes an invalid relationship
        from valid but indistinguishable banks. No configuration is modified.

    Raises
    ------
    TypeError
        A supplied configuration does not have the required tuple type.
    ValueError
        A leaf shape or the source/resolved batch layouts disagree.

    Notes
    -----
    This internal numerical helper supports jit and reads no host values. Callers
    must validate physical configurations separately through Core. A source match
    alone establishes neither physical validity nor recorded map provenance.
    """
    active_shape = jnp.shape(env_config.agent_profile.active_mask)
    num_envs = active_shape[0] if len(active_shape) == 2 else None
    batched = _config_has_batch(env_config, num_envs)
    _config_has_batch(source_config, num_envs)
    prefix = active_shape[:1] if batched else ()
    unchanged = jnp.ones(prefix, dtype=jnp.bool_)
    for name in EnvConfig._fields:
        if name == "team_spawn_pad_positions":
            continue
        for value, source in zip(
            jax.tree.leaves(getattr(env_config, name)),
            jax.tree.leaves(getattr(source_config, name)),
            strict=True,
        ):
            equal = jnp.asarray(value) == jnp.asarray(source)
            unchanged &= jnp.all(equal, axis=tuple(range(int(batched), equal.ndim)))
    pads = env_config.team_spawn_pad_positions
    source_order = jnp.all(
        pads == source_config.team_spawn_pad_positions, axis=(-3, -2, -1)
    )
    exchanged_order = jnp.all(
        pads == _swap_spawn_banks(source_config).team_spawn_pad_positions,
        axis=(-3, -2, -1),
    )
    matches_source = unchanged & (source_order | exchanged_order)
    spawn_choice = jnp.where(
        matches_source & (source_order ^ exchanged_order),
        jnp.where(source_order, 0, 1),
        -1,
    ).astype(jnp.int32)
    return matches_source, spawn_choice


def _source_config_with_class_ids(  # pyright: ignore[reportUnusedFunction]
    source: EnvConfig, source_class_ids: Array
) -> tuple[EnvConfig, Array]:
    """Rebuild a declared source roster from Core's class catalog.

    Parameters
    ----------
    source : EnvConfig
        Scalar source or native batch. Its profile sets the expected lane shape.
        Physical validation belongs to setup; this helper changes no other field.
    source_class_ids : integer array
        Same-leading-shape ten-slot declaration. Each team's nonzero classes
        1..5 must precede zero padding. Repeated classes and empty teams are
        allowed. Ten -1 values retain the complete original profile, including
        its capability values. This helper does not broadcast declarations.

    Returns
    -------
    tuple[EnvConfig, Array]
        Rebuilt source and Boolean validity with the source's lane shape.
        Inputs stay unchanged. A malformed traced row retains the original
        profile and returns False; the caller must keep that failure evidence.

    Raises
    ------
    TypeError
        Source types or declaration integer dtype are invalid.
    ValueError
        Shapes differ or a concrete row has invalid classes or padding. Values
        are checked before int32 narrowing and before catalog indexing.

    Notes
    -----
    Supports jit and external vmap. It uses Core's resolver, not copied class
    tables. Training's no-duplicate, canonical order and Priest rules belong to
    the sampler; generic recording declarations need only compact valid rows.
    """
    shape = jnp.shape(source.agent_profile.active_mask)
    batch = shape[0] if len(shape) == 2 else None
    _config_has_batch(source, batch)
    if isinstance(source_class_ids, Tracer):
        values = cast(Array, source_class_ids)
        if not jnp.issubdtype(values.dtype, jnp.integer) or jnp.issubdtype(
            values.dtype, jnp.bool_
        ):
            raise TypeError("source_class_ids must contain integers")
    else:
        raw = np.asarray(source_class_ids)
        if raw.dtype.kind not in "iu":
            raise TypeError("source_class_ids must contain integers")
        if raw.shape != shape:
            raise ValueError(f"source_class_ids must have shape {shape}")
        sentinel = np.all(raw == -1, axis=-1) if raw.dtype.kind == "i" else False
        teams = raw.reshape((*shape[:-1], NUM_TEAMS, MAX_AGENTS_PER_TEAM))
        active = teams > 0
        compact = np.all(
            active == (np.arange(MAX_AGENTS_PER_TEAM) < active.sum(-1)[..., None]),
            axis=(-2, -1),
        )
        in_range = np.all(raw < NUM_CLASSES, axis=-1)
        if raw.dtype.kind == "i":
            in_range &= np.all(raw >= NEUTRAL_CLASS_ID, axis=-1)
        if not np.all(sentinel | (in_range & compact)):
            raise ValueError(
                "source_class_ids require compact classes 0..5 or ten -1 values"
            )
        values = jnp.asarray(raw.astype(np.int32))
    if values.shape != shape:
        raise ValueError(f"source_class_ids must have shape {shape}")
    sentinel = (
        jnp.all(values == -1, axis=-1)
        if jnp.issubdtype(values.dtype, jnp.signedinteger)
        else jnp.zeros(shape[:-1], dtype=jnp.bool_)
    )
    teams = values.reshape((*shape[:-1], NUM_TEAMS, MAX_AGENTS_PER_TEAM))
    active = teams > 0
    sizes = jnp.sum(active, axis=-1, dtype=jnp.int32)
    compact = jnp.all(
        active == (jnp.arange(MAX_AGENTS_PER_TEAM) < sizes[..., None]),
        axis=(-2, -1),
    )
    valid = sentinel | (
        jnp.all((values >= NEUTRAL_CLASS_ID) & (values < NUM_CLASSES), axis=-1)
        & compact
    )
    replace_profile = valid & ~sentinel
    safe = jnp.where(replace_profile[..., None], values, 0).astype(jnp.int32)
    sizes = jnp.where(replace_profile[..., None], sizes, 0)
    profile = (
        resolve_agent_profile(safe, sizes)
        if batch is None
        else jax.vmap(resolve_agent_profile)(safe, sizes)
    )

    def choose_profile(new: Array, old: Array) -> Array:
        """Retain original profile rows for absent or invalid declarations."""
        return jnp.where(replace_profile[..., None], new, old)

    profile = jax.tree.map(choose_profile, profile, source.agent_profile)
    return source._replace(agent_profile=profile), valid


def _map_info(map_id: int) -> TDMMapInfo:
    """Return the current catalog entry for a plain Python map ID.

    map_id must be an int from 0 through 51; booleans and non-integers are rejected
    with ValueError. The catalog owns names and source identities.
    """
    if type(map_id) is not int or not 0 <= map_id < len(list_tdm_maps()):
        raise ValueError("map_id must be an approved integer from 0 through 51")
    return list_tdm_maps()[map_id]


def _roster_ids(roster: Sequence[AgentClassName], *, name: str) -> tuple[int, ...]:
    """Convert ordered class names to five team-local class IDs.

    roster is a non-string sequence of one through five supported class names.
    Repeated classes remain separate agents in their supplied order. Append
    NEUTRAL_CLASS_ID for unused slots. name identifies the team in errors.
    Unsupported containers raise TypeError; invalid length or class names raise
    ValueError. No placement or class-based reordering happens here.
    """
    if isinstance(roster, (str, bytes)) or not isinstance(
        cast(object, roster), Sequence
    ):
        raise TypeError(f"{name} must be an ordered sequence of class names")
    if not 1 <= len(roster) <= MAX_AGENTS_PER_TEAM:
        raise ValueError(f"{name} must contain one through five agents")
    for value in roster:
        if type(value) is not str or value not in _CLASS_IDS:
            raise ValueError(f"{name} has unsupported class {value!r}")
    return tuple(_CLASS_IDS[value] for value in roster) + (NEUTRAL_CLASS_ID,) * (
        MAX_AGENTS_PER_TEAM - len(roster)
    )


def _make_config(
    geometry: MapGeometry,
    class_ids: tuple[int, ...],
    team_sizes: tuple[int, int],
    *,
    score_threshold: int,
    red_zone_depth: float,
    max_steps: int,
    movement_scale: float = float(CANONICAL_PRODUCT_MOVEMENT_SCALE),
    shield_duration: int = 3,
    shield_speed: float = 2.0,
    wave_periods: tuple[int, int] = (5, 5),
) -> EnvConfig:
    """Combine map geometry, resolved roster choices and TDM rules at host setup.

    Parameters
    ----------
    geometry : MapGeometry
        Loaded map bounds, obstacles and both complete ordered spawn banks.
    class_ids : tuple[int, ...]
        Ten class IDs in Team A then Team B order, including inactive padding.
    team_sizes : tuple[int, int]
        Active counts for Team A and Team B.
    score_threshold : int
        Required TDM winning score threshold.
    red_zone_depth : float
        Required Red Zone depth in map units (EnvConfig field
        team_deathmatch_red_zone_depth). 0.0 keeps one point per death.
    max_steps : int
        Required episode step limit.
    movement_scale : float, optional
        Ordinary movement multiplier. Defaults to CANONICAL_PRODUCT_MOVEMENT_SCALE;
        the product validator requires that calibrated value.
    shield_duration : int, default=3
        Configured spawn-shield duration in transitions.
    shield_speed : float, default=2.0
        Movement speed while a shield is active, in world units per transition.
    wave_periods : tuple[int, int], default=(5, 5)
        Respawn-wave periods for Team A and Team B, in transitions.

    Returns
    -------
    EnvConfig
        A scalar configuration with resolved class facts and float32/int32 arrays.

    Notes
    -----
    resolve_agent_profile and validate_product_env_config own roster and physical
    checks; their errors propagate. No scenario start or random key is created.
    Ordinary Core reset and later respawns use the supplied pad order.
    """
    config = EnvConfig(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=score_threshold,
        team_deathmatch_red_zone_depth=red_zone_depth,
        max_steps=max_steps,
        map_width=geometry.map_width,
        map_height=geometry.map_height,
        obstacles=jnp.asarray(geometry.obstacles, dtype=jnp.float32),
        agent_profile=resolve_agent_profile(
            jnp.asarray(class_ids, dtype=jnp.int32),
            jnp.asarray(team_sizes, dtype=jnp.int32),
        ),
        ordinary_movement_distance_scale=movement_scale,
        team_spawn_pad_positions=jnp.asarray(
            geometry.team_spawn_pad_positions, dtype=jnp.float32
        ),
        spawn_shield_duration_steps=shield_duration,
        spawn_shield_movement_speed=shield_speed,
        team_respawn_wave_period_step_count=jnp.asarray(wave_periods, dtype=jnp.int32),
    )
    validate_product_env_config(config)
    return config


def make_standard_team_deathmatch_config(
    *,
    map_id: int,
    team_a_roster: Sequence[AgentClassName],
    team_b_roster: Sequence[AgentClassName],
    score_threshold: int = 20,
    max_steps: int = 300,
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH,
) -> EnvConfig:
    """Build TDM on an installed map with the supplied roster order.

    Parameters
    ----------
    map_id : int
        Installed map ID from 0 through 51. Use list_tdm_maps to inspect choices.
    team_a_roster, team_b_roster : sequence of str
        Each team's ordered one-to-five class names: mage, warrior, hunter,
        rogue or priest. Repeated classes are separate agents. Teams may have
        different sizes. A string alone is not a roster sequence.
    score_threshold : int, default=20
        Winning score threshold, subject to Core's TDM bounds.
    max_steps : int, default=300
        Episode step limit, subject to Core's valid range.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        How far each team's Red Zone reaches in from its own spawn edge, in
        map units. When an agent dies inside its own team's Red Zone, the
        enemy team gets 2 points instead of 1; it is still one kill and one
        death. 0.0 turns the rule off. Must be a Python float that Core
        accepts (at most the map width after float32 conversion).

    Returns
    -------
    EnvConfig
        Scalar configuration with exact source spawn banks. Supplied class order
        fills each team's first slots; remaining slots are inactive. Movement uses
        the product calibration, respawn waves use five transitions, and the
        configured spawn shield lasts three transitions.

    Raises
    ------
    TypeError
        A roster container or configuration scalar has an unsupported type.
    ValueError
        A map ID, roster entry/size, rule value or requested configuration fails
        its existing catalog or Core check.

    Notes
    -----
    This host factory does not balance spawn locations by itself. Use make's
    ready-environment path or balanced_spawn_configs when both banks are wanted.
    It does not run games, draw randomness or modify the installed assets.
    """
    ids = _roster_ids(team_a_roster, name="team_a_roster") + _roster_ids(
        team_b_roster, name="team_b_roster"
    )
    return _make_config(
        map_geometry(_map_info(map_id)),
        ids,
        (len(team_a_roster), len(team_b_roster)),
        score_threshold=score_threshold,
        red_zone_depth=red_zone_depth,
        max_steps=max_steps,
    )


def make_canonical_team_deathmatch_evaluation_config(
    *, map_id: int, red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH
) -> EnvConfig:
    """Build the fixed mirrored 5v5 setup on an installed test map.

    Parameters
    ----------
    map_id : int
        One of the held-out test map IDs 47 through 51.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Red Zone depth in map units; 0.0 keeps one point per death. Official
        canonical releases keep the rules recorded in their snapshots; this
        factory does not change them.

    Returns
    -------
    EnvConfig
        Scalar TDM configuration with mage, warrior, hunter, rogue and priest
        in each team's supplied canonical order, score threshold 20, the
        given Red Zone depth and a 300-transition limit. Uses the source map's
        complete ordered banks.

    Raises
    ------
    ValueError
        map_id is not a plain Python int identifying an allowed test map, or
        existing configuration validation rejects the setup.

    Notes
    -----
    canonical_tournament_rosters owns the shared roster definition. This factory
    does not create a paired schedule, exchange banks or qualify an official run.
    """
    _map_info(map_id)
    if map_id not in CANONICAL_TDM_EVALUATION_MAP_IDS:
        raise ValueError(
            "canonical evaluation map_id must be one of "
            f"{CANONICAL_TDM_EVALUATION_MAP_IDS}"
        )
    team_a, team_b = canonical_tournament_rosters()
    return make_standard_team_deathmatch_config(
        map_id=map_id,
        team_a_roster=team_a,
        team_b_roster=team_b,
        red_zone_depth=red_zone_depth,
    )


def _initial_state_digest(state: EnvState) -> str:
    """Hash all authored state leaves with their dtype, shape and values.

    state is a concrete scalar EnvState. Read its arrays on the host and encode
    the fixed dev-resolved-initial-state@1 payload before canonical hashing.
    Return a SHA-256 hex digest. This may synchronize device work and belongs
    at the loading boundary, not inside a compiled rollout.
    """
    payload: dict[str, object] = {"schema": "dev-resolved-initial-state@1"}
    for field_name, value in zip(state._fields, state, strict=True):
        host = np.asarray(value)
        payload[field_name] = {
            "dtype": str(host.dtype),
            "shape": list(host.shape),
            "values": host.tolist(),
        }
    return canonical_digest_sha256(payload)


def load_tdm_scenario(scenario_id: int) -> TDMScenario:
    """Load and validate one exact packaged scenario start.

    Parameters
    ----------
    scenario_id : int
        Plain Python integer from 1 through 8. Booleans are not accepted.

    Returns
    -------
    TDMScenario
        Matching catalog entry, resolved configuration, authored initial state
        and notes. Authored positions, health, scores and starting step are
        retained. The configuration plays at the scenario's recorded Red Zone
        depth (5.0 for every installed scenario).

    Raises
    ------
    ValueError
        The ID is invalid, configuration/state/horizon disagrees with the package
        identity, or existing scenario validation rejects the authored start.
    TypeError
        A packaged configuration fails Core's type requirements.

    Notes
    -----
    This host operation reads installed package resources and checks their byte
    and semantic identities. File-reading and schema-validation errors propagate.
    It does not open mutable DevClient saves, advance a game or resample a start.
    Prepare the result before entering jit; use the public reset scenario argument
    or Core's scenario initializer to obtain matching decision inputs.
    """
    if type(scenario_id) is not int or not 1 <= scenario_id <= 8:
        raise ValueError("scenario_id must be an approved integer from 1 through 8")
    info = list_tdm_scenarios()[scenario_id - 1]
    content = scenario_content(info)
    return _load_tdm_scenario(info, content)


def _load_tdm_scenario(info: TDMScenarioInfo, content: ScenarioContent) -> TDMScenario:
    """Resolve one scenario whose installed bytes were already verified.

    info is its current catalog entry; content comes from scenario_content(info).
    This shared host path performs the public loader's exact configuration,
    state and horizon checks without reading those bytes again: it rebuilds
    the config at the record's Red Zone depth and requires the resolved config
    V2 digest to equal the recorded one. Return the validated TDMScenario.
    Validation errors propagate; no episode is advanced.
    Internal setup callers own the byte verification and must keep this pair.
    """
    resolved = content.configuration
    geometry = MapGeometry(
        map_width=resolved.map_width,
        map_height=resolved.map_height,
        obstacles=tuple(
            (
                float(row.obstacle_type_id),
                row.x,
                row.y,
                row.radius,
                row.width,
                row.height,
                row.theta,
                float(row.is_active),
            )
            for row in resolved.obstacle_slots
        ),
        team_spawn_pad_positions=resolved.team_spawn_pad_positions,
    )
    config = _make_config(
        geometry,
        info.class_ids,
        info.team_sizes,
        score_threshold=resolved.team_deathmatch_score_threshold,
        red_zone_depth=resolved.team_deathmatch_red_zone_depth,
        max_steps=resolved.maximum_episode_steps,
        movement_scale=resolved.ordinary_movement_distance_scale,
        shield_duration=resolved.spawn_shield_duration_steps,
        shield_speed=resolved.spawn_shield_movement_speed,
        wave_periods=cast(tuple[int, int], resolved.team_respawn_wave_period_steps),
    )
    if (
        build_resolved_env_config_v2(config) != resolved
        or resolved.canonical_digest_sha256 != info.resolved_configuration_digest
    ):
        raise ValueError(
            "packaged scenario configuration disagrees with live mechanics"
        )
    snapshot = content.initial_snapshot.model_dump()
    snapshot.pop("schema_id")
    snapshot.pop("schema_version")
    snapshot["step_count"] = content.step_count
    float_fields = {"agent_positions", "current_health"}
    bool_fields = {"alive_mask", "has_previous_timestep_joint_action"}
    state_values: dict[str, Array] = {}
    for field_name in EnvState._fields:
        dtype = (
            jnp.float32
            if field_name in float_fields
            else jnp.bool_
            if field_name in bool_fields
            else jnp.int32
        )
        state_values[field_name] = jnp.asarray(snapshot[field_name], dtype=dtype)
    state = EnvState(**state_values)
    validate_scenario_initial_state(config, state)
    if _initial_state_digest(state) != info.resolved_initial_state_digest:
        raise ValueError("packaged scenario initial-state digest mismatch")
    if config.max_steps - int(state.step_count) != info.horizon:
        raise ValueError("packaged scenario horizon disagrees with its manifest")
    return TDMScenario(
        info=info, config=config, initial_state=state, notes=content.notes
    )


__all__ = [
    "CANONICAL_TDM_EVALUATION_MAP_IDS",
    "AgentClassName",
    "TDMAssetSource",
    "TDMMapInfo",
    "TDMScenario",
    "TDMScenarioInfo",
    "balanced_spawn_configs",
    "canonical_tournament_rosters",
    "list_tdm_maps",
    "list_tdm_scenarios",
    "load_tdm_scenario",
    "make_canonical_team_deathmatch_evaluation_config",
    "make_standard_team_deathmatch_config",
]
