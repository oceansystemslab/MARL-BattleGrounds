"""Sample approved map and roster distributions with independent lane keys.

Prepare the immutable source bank once with prepare_training_content. Validate
changing map/team-size controls on the host, then pass them as dynamic arrays to
sample_training_configs inside jit or scan. Sampling reads no assets, hashes no
content and stores no random state. The caller adopts results only at reset;
the existing environment remains the sole reset and simulation owner.
"""

from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.core.types import (
    MAX_AGENTS_PER_TEAM,
    NUM_CLASSES,
    PRIEST_CLASS_ID,
    EnvConfig,
)
from marl_battlegrounds.tasks import (
    _config_has_batch,  # pyright: ignore[reportPrivateUsage]
    _roster_ids,  # pyright: ignore[reportPrivateUsage]
    _source_config_with_class_ids,  # pyright: ignore[reportPrivateUsage]
    balanced_spawn_configs,
    canonical_tournament_rosters,
)

TRAINING_KEY_SCHEMA_VERSION = 1
type TrainingStream = Literal[
    "map", "roster", "opponent", "action", "reset", "step", "initialization"
]
_STREAM_TAGS: dict[str, int] = {
    "map": 0,
    "roster": 1,
    "opponent": 2,
    "action": 3,
    "reset": 4,
    "step": 5,
    "initialization": 6,
}
_MAP_COUNT = 42


class SampledTrainingConfigs(NamedTuple):
    """Keep one sampled configuration and its declaration per lane.

    Attributes
    ----------
    config : EnvConfig
        Every leaf begins with B. The first half of lanes use source spawn
        order; the second half exchange both full banks. Pass directly to reset.
    source_indices : Array
        Int32 (B,) rows in the immutable threshold-major source bank. Each
        threshold contributes 42 maps; source_indices % 42 gives map IDs.
    source_class_ids : Array
        Int32 (B,10) compact Team A then Team B class declarations, with zero
        padding. These describe the source profile before spawn balancing.

    Notes
    -----
    A numerical PyTree suitable for jit/scan. Inputs are unchanged. Adopt these
    values only for lanes that reset; do not retain a full history of configs.
    """

    config: EnvConfig
    source_indices: Array
    source_class_ids: Array


def _control_shapes(eligible_maps: Array, team_size: Array) -> None:
    """Check dynamic control shapes and dtypes without reading traced values.

    eligible_maps must be bool (42,) and team_size scalar int32. Raise TypeError
    for wrong dtypes and ValueError for wrong shapes. Numerical bounds are the
    host validator's responsibility.
    """
    if np.shape(eligible_maps) != (_MAP_COUNT,):
        raise ValueError("eligible_maps must have shape (42,)")
    if getattr(eligible_maps, "dtype", None) != jnp.bool_:
        raise TypeError("eligible_maps must have Boolean dtype")
    if np.shape(team_size) != ():
        raise ValueError("team_size must be a scalar")
    if getattr(team_size, "dtype", None) != jnp.int32:
        raise TypeError("team_size must have int32 dtype")


def validate_training_distribution(
    *, eligible_maps: Array, team_size: Array, roster_class_ids: Array | None = None
) -> None:
    """Check map eligibility and equal team size before compiled sampling.

    Parameters
    ----------
    eligible_maps : Array
        Boolean (42,) mask in the prepared bank's source order. At least one
        True is required. This function does not inspect content eligibility.
    team_size : Array
        Scalar int32 in 1..5, used by both teams. Boolean and float values fail.
    roster_class_ids : Array or None, default=None
        Optional int32 (10,) explicit Team A then Team B roster. Each team has
        one through five classes 1..5 followed by zero padding. Repeats and
        unequal team sizes are allowed. An all-zero row keeps team_size sampling.

    Returns
    -------
    None
        Controls are valid and unchanged. Validate each new distribution on the
        host, then pass the same-shaped controls dynamically into compiled calls.

    Raises
    ------
    TypeError
        Dtypes are wrong, or called with tracers inside jit, scan or vmap.
    ValueError
        Shapes are wrong, no map is eligible, or team size is outside 1..5.

    Notes
    -----
    Host-only; reading device controls can synchronize and transfer 43 values,
    plus ten when an explicit roster control is supplied.
    It reads no files and writes no state. No curriculum schedule is implied.
    """
    _control_shapes(eligible_maps, team_size)
    if isinstance(eligible_maps, Tracer) or isinstance(team_size, Tracer):
        raise TypeError("Validate training controls on the host before compiled use")
    if not bool(np.any(np.asarray(eligible_maps))):
        raise ValueError("eligible_maps must contain at least one eligible map")
    if not 1 <= int(np.asarray(team_size)) <= MAX_AGENTS_PER_TEAM:
        raise ValueError("team_size must be in 1..5")
    if roster_class_ids is not None:
        _roster_shape(roster_class_ids)
        if isinstance(roster_class_ids, Tracer):
            raise TypeError("Validate roster controls on the host before compiled use")
        rows = np.asarray(roster_class_ids).reshape(2, MAX_AGENTS_PER_TEAM)
        if not np.any(rows):
            return
        active = rows > 0
        sizes = active.sum(axis=-1)
        if (
            np.any(rows < 0)
            or np.any(rows >= NUM_CLASSES)
            or np.any(sizes == 0)
            or not np.array_equal(
                active, np.arange(MAX_AGENTS_PER_TEAM) < sizes[:, None]
            )
        ):
            raise ValueError("Explicit rosters need compact nonempty classes 1..5")


def _roster_shape(roster_class_ids: Array) -> None:
    """Check the static ten-slot roster contract without reading array values."""
    if np.shape(roster_class_ids) != (2 * MAX_AGENTS_PER_TEAM,):
        raise ValueError("roster_class_ids must have shape (10,)")
    if getattr(roster_class_ids, "dtype", None) != jnp.int32:
        raise TypeError("roster_class_ids must have int32 dtype")


def _counter(value: Array, *, name: str, shape: tuple[int, ...] | None = None) -> None:
    """Check an int32 lane counter; concrete values must be nonnegative.

    value has one nonempty lane axis, or exactly shape when supplied. Traced
    numerical bounds are a caller precondition; no host callback is inserted.
    Wrong dtypes raise TypeError and invalid shapes/values raise ValueError.
    """
    if getattr(value, "dtype", None) != jnp.int32:
        raise TypeError(f"{name} must have int32 dtype")
    if (
        value.ndim != 1
        or not value.shape[0]
        or (shape is not None and value.shape != shape)
    ):
        raise ValueError(f"{name} must have a matching nonempty lane axis")
    if not isinstance(value, Tracer) and np.any(np.asarray(value) < 0):
        raise ValueError(f"{name} must be nonnegative")


def training_keys(
    root_key: Array,
    reset_generation: Array,
    *,
    stream: TrainingStream,
    decision_step: Array | None = None,
) -> Array:
    """Derive version-1 keys independently for each fixed environment lane.

    Parameters
    ----------
    root_key : Array
        Scalar typed Threefry key, or legacy uint32 (2,) key with the default
        Threefry implementation. RBG and batched roots are rejected.
    reset_generation : Array
        Nonnegative int32 (B,) counters. Initial episodes use zero; advance only
        reset lanes with the existing overflow-checked lifecycle authority.
    stream : str
        Static domain: map=0, roster=1, opponent=2, action=3, reset=4, step=5,
        initialization=6. Opponent is reserved; no opponent is selected here.
    decision_step : Array | None
        Required nonnegative int32 (B,) local decision indices for action and
        step streams. Must be None for every other stream.

    Returns
    -------
    Array
        Typed keys (B,) or legacy uint32 (B,2), preserving the root format.
        Fold order is root, domain, fixed lane index, generation, then decision
        step when required. Another lane's resets cannot change these keys.

    Raises
    ------
    TypeError, ValueError
        Key format, stream, counter shapes/dtypes or concrete values are invalid.

    Notes
    -----
    Supports jit/vmap with dynamic keys/counters and static stream. Traced
    counters must already satisfy numerical bounds. Save root bits, Threefry
    implementation, schema version, batch/lane order and counters for resume.
    Pass action keys directly to M8, which owns team and actor key derivation.
    Arbitrary stochastic System initialization may also use M8 episode IDs;
    this helper does not promise identical initialization across reset schedules.
    """
    if stream not in _STREAM_TAGS:
        raise ValueError("Unknown training random-key stream")
    if str(jax.random.key_impl(root_key)) != "threefry2x32":
        raise ValueError("Training sampling requires Threefry random keys")
    if jax.random.key_data(root_key).shape != (2,):
        raise ValueError("root_key must be one scalar Threefry key")
    _counter(reset_generation, name="reset_generation")
    needs_step = stream in ("action", "step")
    if needs_step != (decision_step is not None):
        raise ValueError("Only action and step streams require decision_step")
    domain = jax.random.fold_in(root_key, _STREAM_TAGS[stream])
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        domain, jnp.arange(reset_generation.shape[0], dtype=jnp.int32)
    )
    keys = jax.vmap(jax.random.fold_in)(keys, reset_generation)
    if decision_step is not None:
        _counter(decision_step, name="decision_step", shape=reset_generation.shape)
        keys = jax.vmap(jax.random.fold_in)(keys, decision_step)
    return keys


def _team_classes(key: Array, canonical: Array, team_size: Array) -> Array:
    """Draw one uniform subset and return its canonical compact five-slot row.

    key is one Threefry team key, canonical int32 (5,) follows the existing
    roster authority and team_size is a valid scalar int32. A uniform random
    permutation gives every eligible subset equal probability. Move Priest to
    the end for size one, select a prefix, then restore canonical relative order.
    Return int32 (5,), with zero padding; no input is changed.
    """
    order = jax.random.permutation(key, MAX_AGENTS_PER_TEAM)
    excluded = (team_size == 1) & (canonical[order] == PRIEST_CLASS_ID)
    order = order[jnp.argsort(excluded, stable=True)]
    selected = jnp.sort(
        jnp.where(
            jnp.arange(MAX_AGENTS_PER_TEAM) < team_size, order, MAX_AGENTS_PER_TEAM
        )
    )
    return jnp.where(
        selected < MAX_AGENTS_PER_TEAM,
        canonical[jnp.minimum(selected, MAX_AGENTS_PER_TEAM - 1)],
        0,
    ).astype(jnp.int32)


def sample_training_configs(
    bank: EnvConfig,
    root_key: Array,
    reset_generation: Array,
    *,
    eligible_maps: Array,
    team_size: Array,
    score_threshold: Array | None = None,
    roster_class_ids: Array | None = None,
) -> SampledTrainingConfigs:
    """Sample approved maps and independent equal-size teams for reset.

    Parameters
    ----------
    bank : EnvConfig
        Immutable threshold-major bank with 42 rows per declared score, prepared
        by prepare_training_content. Both spawn orders must be valid.
    root_key : Array
        Scalar typed or legacy Threefry root; see training_keys for ownership.
    reset_generation : Array
        Nonnegative int32 (B,), with fixed positive even B. Supply each lane's
        intended next generation when sampling its next episode.
    eligible_maps : Array
        Nonempty Boolean (42,) map mask. Each eligible map is equally likely.
    team_size : Array
        Scalar int32 in 1..5. Both teams draw independently without replacement;
        Priest is excluded only at size one. Five uses exact canonical rosters.
    score_threshold : Array or None, default=None
        Scalar int32 score present in the prepared bank. None selects K20.
        This changes only the selected source block, not map/roster random keys.
        Validate concrete values before passing them through jit or scan.
    roster_class_ids : Array or None, default=None
        Optional int32 (10,) ordered roster, with Team A then Team B classes
        and zero padding. A nonzero row replaces random roster selection and
        may contain repeated classes or unequal team sizes. None or an all-zero
        row retains the existing team_size distribution and its random keys.

    Returns
    -------
    SampledTrainingConfigs
        Batched configurations, source indices and compact class declarations.
        Only profiles and the fixed second-half spawn exchange differ from the
        selected source. Team identity and pad order are preserved.

    Raises
    ------
    TypeError, ValueError
        Static shapes/dtypes/keys or concrete numerical controls are invalid.

    Notes
    -----
    Supports jit/vmap and changing same-shaped values without new compilation.
    Validate each dynamic distribution with validate_training_distribution on
    the host first. Traced invalid numerical controls are unsupported; no repair
    or fallback is provided. The bank is setup-validated, not rehashed here.
    Gate this call and reset_done with one condition over any reset lane. Adopt
    outputs only for reset lanes; continuing configurations/declarations stay
    unchanged. Never balance this output a second time. This is not a curriculum
    or a training loop. It needs base dependencies only.
    """
    _control_shapes(eligible_maps, team_size)
    if roster_class_ids is not None:
        _roster_shape(roster_class_ids)
    threshold = (
        jnp.asarray(20, jnp.int32) if score_threshold is None else score_threshold
    )
    if np.shape(threshold) != () or getattr(threshold, "dtype", None) != jnp.int32:
        raise TypeError("score_threshold must be scalar int32")
    if not any(
        isinstance(value, Tracer)
        for value in (eligible_maps, team_size, roster_class_ids)
    ):
        validate_training_distribution(
            eligible_maps=eligible_maps,
            team_size=team_size,
            roster_class_ids=roster_class_ids,
        )
    _counter(reset_generation, name="reset_generation")
    batch = reset_generation.shape[0]
    if batch % 2:
        raise ValueError("Training sampling requires a positive even batch")
    count = np.shape(bank.agent_profile.active_mask)[0]
    if count < _MAP_COUNT or count % _MAP_COUNT or not _config_has_batch(bank, count):
        raise ValueError("bank must have 42 source rows per threshold")
    scores = jnp.asarray(bank.team_deathmatch_score_threshold)[::_MAP_COUNT]
    if (
        not isinstance(threshold, Tracer)
        and not isinstance(scores, Tracer)
        and int(np.asarray(threshold)) not in np.asarray(scores)
    ):
        raise ValueError("score_threshold is not present in the prepared bank")
    block = jnp.argmax(scores == threshold).astype(jnp.int32)
    keys = training_keys(root_key, reset_generation, stream="map")
    logits = jnp.where(eligible_maps, jnp.float32(0), jnp.float32(-jnp.inf))

    def draw_map(key: Array) -> Array:
        """Select one uniformly eligible source with the lane's map key."""
        return jax.random.categorical(key, logits).astype(jnp.int32)

    indices = jax.vmap(draw_map)(keys) + block * _MAP_COUNT

    def select(value: Array) -> Array:
        """Gather one source value per lane without copying the complete bank."""
        return value[indices]

    selected = jax.tree.map(select, bank)
    rosters = canonical_tournament_rosters()
    canonical = jnp.asarray(
        [_roster_ids(roster, name="canonical roster") for roster in rosters],
        dtype=jnp.int32,
    )

    def draw_rosters() -> Array:
        """Draw each team's roster beneath its independent lane/domain key."""
        lane_keys = training_keys(root_key, reset_generation, stream="roster")

        def draw_lane(key: Array) -> Array:
            """Return one ten-slot row from separate Team A and Team B tags."""
            return jnp.concatenate(
                [
                    _team_classes(
                        jax.random.fold_in(key, team), canonical[team], team_size
                    )
                    for team in range(2)
                ]
            )

        return jax.vmap(draw_lane)(lane_keys)

    def sampled_rosters() -> Array:
        """Keep the original random/canonical roster rule and key ownership."""
        return cast(
            Array,
            jax.lax.cond(
                team_size == MAX_AGENTS_PER_TEAM,
                lambda: jnp.broadcast_to(canonical.reshape(10), (batch, 10)),
                draw_rosters,
            ),
        )

    classes = cast(
        Array,
        sampled_rosters()
        if roster_class_ids is None
        else jax.lax.cond(
            jnp.any(roster_class_ids != 0),
            lambda: jnp.broadcast_to(roster_class_ids, (batch, 10)),
            sampled_rosters,
        ),
    )
    selected, _ = _source_config_with_class_ids(selected, classes)
    config = balanced_spawn_configs(selected, num_envs=batch)
    return SampledTrainingConfigs(config, indices, classes)
