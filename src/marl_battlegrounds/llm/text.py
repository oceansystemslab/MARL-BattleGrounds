"""Render one permitted actor input as compact, exact, readable game text.

format_actor_view is a host-only presentation helper. It never reads Core state,
configuration, another actor's private input or a provider. Its caller chooses
and transforms the coordinate frame before calling it. Equal float32 rows share
one named definition; their distinct visibility and source identities remain.
"""

from typing import Literal, cast

import jax
import numpy as np
import numpy.typing as npt

from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.core.combat import GLOBAL_SLOW_FLOOR
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.llm.actions import MOVE_NAMES, TARGET_NAMES, legal_action_names
from marl_battlegrounds.policies.input import ActorInput

TEXT_VERSION = "actor-text-v1"
type _HostArray = npt.NDArray[np.generic]


def _column_names(prefix: str, size: int) -> tuple[str, ...]:
    """Read ordered field names from Core constants; fail on a changed schema."""
    names = {
        value: name.removeprefix(prefix).lower()
        for name, value in vars(core_types).items()
        if name.startswith(prefix) and isinstance(value, int)
    }
    if set(names) != set(range(size)):
        raise RuntimeError(f"Core's {prefix} schema needs a text-version update")
    return tuple(names[index] for index in range(size))


_UNIT_FIELDS = _column_names("AGENT_FEATURE_", core_types.UNIT_FEATURES)
_CONTEXT_FIELDS = _column_names("CONTEXT_FEATURE_", core_types.CONTEXT_FEATURES)
_OBSTACLE_FIELDS = _column_names("OBSTACLE_FEATURE_", core_types.OBSTACLE_FEATURES)
_RULES = (
    "Rules: Choose one move and one complete combat pair from the legal lists. "
    "All actors commit both choices from the same decision. Damage and healing "
    "are summed before health clipping and death. Combat reads starting positions and "
    "statuses; Charge relocation precedes the chosen ordinary movement. New "
    "statuses first control the next decision. Legal choices can have no effect "
    "because of game physics. Previous actions describe accepted choices, not hits.\n"
    "Abilities: Basic and Ultimate are alternatives. Priest Basic heals an ally, "
    "including self, and grants a temporary speed floor. Other Basics damage an "
    "enemy; Hunter Basic also slows it. Mage Ultimate gives self Burst damage "
    "amplification without a target. Warrior Ultimate damages, slows and stuns "
    "an enemy and moves the "
    "Warrior toward it. Hunter Ultimate damages and traps an enemy. Rogue "
    "Ultimate damages and applies slow, stun and reduced healing. Priest "
    "Ultimate heals an ally. capability_* fields describe what a unit can apply; "
    "other status fields describe effects currently on it. Use the displayed "
    "amounts, ranges and durations. Ultimate starts its full cooldown.\n"
    "Sight and control: Direct sight needs a living observer and target, range "
    "and a clear line through static obstacles; units do not block sight. Shared "
    "sight does not expand your legal combat choices. Shielded enemies are "
    "hidden; shielded units cannot use or receive targeted combat. Stun prevents "
    "movement and combat. Different slows multiply with minimum speed fraction "
    f"{GLOBAL_SLOW_FLOOR}; "
    "Priest's speed floor can raise this. Existing effects age once per tick; "
    "reapplication refreshes durations. Positive damage breaks an existing Hunter "
    "Trap in the next state, but does not remove a Trap first applied that tick. "
    "Poison reduces healing and recovery; it does not damage each tick.\n"
    "Team effects: Mage auras increase outgoing damage; Warrior auras reduce "
    "incoming damage. Living unshielded allies, including the emitter, benefit "
    "within aura range without a sight requirement. Matching auras multiply up "
    "to two emitters' strength. Recovery needs a countdown already at zero and "
    "no combat participation that tick; healing a unit already in combat counts "
    "as participation.\n"
    "Units: Positions, radii, widths and heights are map units; movement speeds "
    "are map units per tick; angles are radians; health, damage and healing use "
    "health units. Durations, cooldowns and clocks count ticks. Multipliers and "
    "fractions are dimensionless. North increases y; east increases x. Diagonal "
    "directions have unit length too. Sight and interaction radii use center-to-"
    "center distance, including their boundary.\n"
    f"Geometry: Obstacle type {core_types.OBSTACLE_TYPE_PILLAR} is a circular pillar "
    f"and {core_types.OBSTACLE_TYPE_WALL} is a rectangular wall. x/y give the center; "
    "radius describes a pillar, and width/height are full wall lengths. theta "
    "rotates a wall counterclockwise from the x axis. Only active obstacles "
    "participate. Bodies have their displayed radius; collisions and map "
    "boundaries can shorten or redirect movement.\n"
    "Roster: Ally/enemy numbers are stable rows 0..4. Class IDs are "
    f"{core_types.MAGE_CLASS_ID}=Mage, {core_types.WARRIOR_CLASS_ID}=Warrior, "
    f"{core_types.HUNTER_CLASS_ID}=Hunter, {core_types.ROGUE_CLASS_ID}=Rogue, "
    f"{core_types.PRIEST_CLASS_ID}=Priest; 0 means unused. "
    "Public alive flags reveal no hidden position. Source permission does not "
    "mean the source currently sees a unit. Respawn clock zero means a wave is "
    "due at transition end; a dead actor cannot act before it respawns.\n"
    "Scoring: In Team Deathmatch a death gives the opponent "
    f"{core_types.TEAM_DEATHMATCH_POINTS_PER_DEATH} point, or "
    f"{core_types.TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH} in the victim's own "
    "Red Zone, using its starting position. The zone is the full-height strip "
    "beside that team's spawn edge, including its boundary. Depth is in context; "
    "zero disables it. "
    "A team wins only when it reaches the threshold with more points than the "
    "opponent. A simultaneous threshold tie, or the horizon without a threshold "
    "winner, is a draw.\n"
    "Encoding: Unlisted numeric fields and padded rows are positive zero. "
    "Zero means a whole positive-zero row. Equal unit definitions share an ID; "
    "visibility and source assignments remain separate. Current objectives have "
    "8 rows of 12 columns; unexpected nonzero reserved values are printed. "
    "Hidden means no direct visible unit row; Not seen means no previous "
    "accepted action was supplied. All numbers preserve their original values."
)


def _number(value: np.generic) -> str:
    """Spell a finite scalar exactly, retaining float32 precision and signed zero."""
    if value.dtype.kind == "b":
        return "1" if bool(value) else "0"
    if value.dtype.kind in "iu":
        return str(int(value))
    if value.dtype != np.dtype(np.float32) or not np.isfinite(value):
        raise ValueError("Actor text requires finite float32 feature values")
    return min(
        np.format_float_positional(cast(np.float32, value), unique=True, trim="-"),
        np.format_float_scientific(
            cast(np.float32, value), unique=True, trim="-", exp_digits=1
        ),
        key=len,
    )


def _nonzero(value: _HostArray) -> _HostArray:
    """Locate values that cannot be represented by the positive-zero default."""
    return (value != 0) | np.signbit(value)


def _fields(row: _HostArray, names: tuple[str, ...]) -> str:
    """Print one fixed-size float32 row, omitting only positive-zero fields."""
    return (
        " ".join(
            f"{names[index]}={_number(row[index])}"
            for index in np.flatnonzero(_nonzero(row))
        )
        or "Zero"
    )


def _array_text(value: _HostArray) -> str:
    """Print a small host scalar or array with exact leaves and bracketed axes."""
    if value.ndim == 0:
        return _number(cast(np.generic, value[()]))
    return "[" + ",".join(_array_text(row) for row in value) + "]"


def _host_actor(actor: ActorInput) -> ActorInput:
    """Transfer and validate one actor's complete shapes, dtypes and finite facts.

    The caller must supply an already permitted ActorInput. This checks the data
    layout, not information rights, actor identity or decision freshness. Batched
    inputs fail instead of accidentally exposing another actor's private row.
    """
    if not isinstance(actor, ActorInput):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Input must be one permitted ActorInput")
    host = jax.device_get(actor)
    observation = host.observation
    expected: list[tuple[object, tuple[int, ...], str]] = [
        (observation.self_features, (58,), "float32"),
        (observation.ally_unit_features, (5, 58), "float32"),
        (observation.enemy_unit_features, (5, 58), "float32"),
        (observation.map_obstacle_features, (32, 8), "float32"),
        (observation.objective_features, (8, 12), "float32"),
        (observation.context_features, (20,), "float32"),
        (observation.ally_visibility_mask, (5,), "bool"),
        (observation.enemy_visibility_mask, (5,), "bool"),
        (observation.self_ally_index, (), "int32"),
        (host.source_availability, (5,), "bool"),
        (
            host.source_bank.unit_features_by_source_and_candidate,
            (5, 10, 58),
            "float32",
        ),
        (host.source_bank.unit_visibility_by_source_and_candidate, (5, 10), "bool"),
        (host.source_bank.objective_features_by_source, (5, 8, 12), "float32"),
    ]
    expected.extend(
        (value, (5, size), "float32")
        for value, size in zip(
            observation.previous_timestep_actions, (9, 9, 11, 11, 2, 2), strict=True
        )
    )
    expected.extend(
        (value, shape, dtype)
        for value, (shape, dtype) in zip(
            observation.spawn_lifecycle,
            (
                ((2, 5, 2), "float32"),
                ((2, 5), "int32"),
                ((), "int32"),
                ((), "float32"),
                ((2,), "int32"),
                ((2,), "int32"),
                ((2, 5), "bool"),
                ((2, 5), "bool"),
                ((2, 5), "int32"),
            ),
            strict=True,
        )
    )
    for value, shape, dtype in expected:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"Actor field must have shape {shape} and dtype {dtype}; "
                f"got {array.shape}, {array.dtype}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("Actor features must be finite")
    return host


def _previous(row: _HostArray, names: tuple[str, ...]) -> str:
    """Name a valid one-hot previous action; preserve any other supplied values."""
    indices = np.flatnonzero(row)
    if not np.any(_nonzero(row)):
        return "Not seen"
    if len(indices) == 1 and row[indices[0]] == 1 and not np.any(np.signbit(row)):
        return names[int(indices[0])]
    return "Values" + _array_text(row)


def _objectives(value: _HostArray) -> str:
    """Print nonzero reserved objective rows with their exact row/column positions."""
    return (
        "; ".join(
            f"{index}:{_array_text(row)}"
            for index, row in enumerate(value)
            if np.any(_nonzero(row))
        )
        or "Zero"
    )


def format_actor_view(
    actor: ActorInput,
    masks: ActionMask,
    *,
    frame: Literal["left", "world"] = "left",
) -> str:
    """Render one permitted current actor view and its complete legal menu.

    Parameters
    ----------
    actor : ActorInput
        One already-filtered actor: self (58,), unit tables (5, 58), obstacles
        (32, 8), context (20,), objectives (8, 12), previous-action tables,
        public spawn lifecycle, source rows (5, 10, 58) and permissions (5,).
        Features are finite float32; masks are Boolean; lifecycle counters and
        self row are int32. No leading batch or actor axis is accepted.
    masks : ActionMask
        Matching actor masks in the same decision epoch and coordinate frame.
    frame : {"left", "world"}, default="left"
        Label of the supplied coordinates, not a request to transform them.
        For left-spawn input, call mirror_team_view with team_on_right first.
        A subsequent action must be mapped back exactly once with mirror_move.

    Returns
    -------
    str
        Deterministic plain text: public rules/map first, then current permitted
        facts, source provenance, previous actions and legal named choices.
        Short decimals round-trip to the original float32 including signed zero.
        Source permission and visibility are never combined. Inputs are unchanged.

    Raises
    ------
    ValueError
        Frame, scalar actor shapes, dtypes, finite values or masks are invalid.

    Notes
    -----
    Host-only. Each input tree is transferred with device_get before formatting;
    pass NumPy leaves to reuse an earlier batched host transfer. There is no
    provider, persistent cache, tactical selection or private-state access.
    Information filtering and matching actor/epoch/frame are caller obligations.
    Default full-field text is optional; custom methods may format allowed data
    differently while keeping the same action checks.
    """
    if frame not in ("left", "world"):
        raise ValueError("Frame must be left or world")
    host = _host_actor(actor)
    menu = legal_action_names(masks)
    obs = host.observation
    lines = [f"MARL-BGs {TEXT_VERSION}; Frame: {frame}", _RULES, "Map:"]
    obstacles = np.asarray(obs.map_obstacle_features)
    lines.extend(
        f"Obstacle {index}: {_fields(row, _OBSTACLE_FIELDS)}"
        for index, row in enumerate(obstacles)
        if np.any(_nonzero(row))
    )
    lines.extend(
        [
            f"Context: {_fields(np.asarray(obs.context_features), _CONTEXT_FIELDS)}",
            f"Self row: ally_{int(obs.self_ally_index)}",
            f"Objectives: {_objectives(np.asarray(obs.objective_features))}",
        ]
    )
    definitions: list[str] = []
    ids: dict[bytes, str] = {}

    def unit(row: _HostArray) -> str:
        """Assign an exact-row ID within this one prompt, without a shared cache."""
        if not np.any(_nonzero(row)):
            return "Zero"
        key = row.tobytes()
        if key not in ids:
            name = f"unit_{len(ids)}"
            ids[key] = name
            definitions.append(f"{name}: {_fields(row, _UNIT_FIELDS)}")
        return ids[key]

    assignments = [f"Self: {unit(np.asarray(obs.self_features))}"]
    for relation, rows, visibility in (
        ("ally", obs.ally_unit_features, obs.ally_visibility_mask),
        ("enemy", obs.enemy_unit_features, obs.enemy_visibility_mask),
    ):
        for index, row in enumerate(np.asarray(rows)):
            visible = "Visible" if visibility[index] else "Hidden"
            assignments.append(f"{relation}_{index}: {visible} {unit(row)}")
    bank = host.source_bank
    for source in range(5):
        permission = "Allowed" if host.source_availability[source] else "Unavailable"
        visible = np.asarray(bank.unit_visibility_by_source_and_candidate[source])
        rows = np.asarray(bank.unit_features_by_source_and_candidate[source])
        entries = [
            f"{TARGET_NAMES[candidate + 1]}="
            f"{'Visible' if visible[candidate] else 'Hidden'} {unit(row)}"
            for candidate, row in enumerate(rows)
            if visible[candidate] or np.any(_nonzero(row))
        ]
        assignments.append(
            f"Source ally_{source}: {permission}; "
            + ("; ".join(entries) or "No visible rows")
        )
        source_objectives = _objectives(
            np.asarray(bank.objective_features_by_source[source])
        )
        assignments.append(f"Source ally_{source} objectives: {source_objectives}")
    lines.extend(["Units (unlisted fields are zero):", *definitions, *assignments])
    lines.append("Unlisted source candidates: Hidden Zero")
    lifecycle_names = (
        "Spawn pads",
        "Shield ticks",
        "Configured shield ticks",
        "Shield speed",
        "Wave periods",
        "Wave countdowns",
        "Active roster",
        "Alive roster",
        "Class IDs",
    )
    lines.append("Spawn lifecycle (team axes: own team, opponent):")
    lines.extend(
        f"{name}: {_array_text(np.asarray(value))}"
        for name, value in zip(lifecycle_names, obs.spawn_lifecycle, strict=True)
    )
    previous = obs.previous_timestep_actions
    lines.append("Previous accepted actions (targets already use your roster names):")
    for relation, moves, targets, ultimates in (
        ("ally", previous[0], previous[2], previous[4]),
        ("enemy", previous[1], previous[3], previous[5]),
    ):
        for index in range(5):
            values = (
                _previous(np.asarray(moves[index]), MOVE_NAMES),
                _previous(np.asarray(targets[index]), TARGET_NAMES),
                _previous(np.asarray(ultimates[index]), ("No", "Yes")),
            )
            lines.append(
                f"{relation}_{index}: "
                + (
                    "Not seen"
                    if values == ("Not seen",) * 3
                    else f"move={values[0]} target={values[1]} ultimate={values[2]}"
                )
            )
    lines.extend(
        [
            "Legal moves: " + ", ".join(menu["move"]),
            "Legal combat pairs: " + ", ".join(menu["combat"]),
            "Combat: no_combat means no target and no Ultimate; ultimate means an "
            "untargeted Ultimate. Target names keep their roster meaning. The target "
            "and Ultimate marginal masks are the unions of the listed complete pairs.",
            'Reply with exactly {"move":"<legal move>","combat":"<legal combat>"}.',
        ]
    )
    return "\n".join(lines)
