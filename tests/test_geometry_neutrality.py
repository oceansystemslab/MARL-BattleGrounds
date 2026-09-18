"""Check that contact handling does not depend on arbitrary actor ordering."""

from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    project_charge_endpoints_with_geometry,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_NONE,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
)

type GeometryInputs = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]

_move = cast(
    Callable[..., Array],
    jax.jit(
        project_movement_with_geometry,
        static_argnames=(
            "agent_agent_overlap_projection_passes",
            "collision_projection_passes",
            "movement_substeps",
        ),
    ),
)
_charge = cast(
    Callable[..., Array],
    jax.jit(
        project_charge_endpoints_with_geometry,
        static_argnames=(
            "agent_agent_overlap_projection_passes",
            "collision_projection_passes",
        ),
    ),
)


def _scene(
    positions: Sequence[Sequence[float]],
    deltas: Sequence[Sequence[float]] = (),
    obstacles: Sequence[Sequence[float]] = (),
    *,
    radius: float = 0.5,
    width: float = 20.0,
    height: float = 20.0,
) -> GeometryInputs:
    count = len(positions)
    centers = np.zeros((MAX_AGENT_SLOTS, 2), np.float32)
    centers[:count] = positions
    movement = np.zeros_like(centers)
    if deltas:
        movement[: len(deltas)] = deltas
    enabled = np.arange(MAX_AGENT_SLOTS) < count
    rows = np.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), np.float32)
    if obstacles:
        rows[: len(obstacles)] = obstacles
    return cast(
        GeometryInputs,
        tuple(
            jnp.asarray(value)
            for value in (
                centers,
                np.full(MAX_AGENT_SLOTS, radius, np.float32),
                movement,
                enabled,
                enabled,
                np.float32(width),
                np.float32(height),
                rows,
                enabled,
                enabled,
            )
        ),
    )


def _assert_positions(result: Array, expected: Array | np.ndarray) -> None:
    actual = np.asarray(result)
    assert actual.shape == (MAX_AGENT_SLOTS, 2)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=GEOMETRY_TOLERANCE)


def _assert_same_position_bits(result: Array, expected: Array) -> None:
    actual, wanted = np.asarray(result), np.asarray(expected)
    assert actual.shape == wanted.shape == (MAX_AGENT_SLOTS, 2)
    assert actual.dtype == wanted.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ((-10, 0), (0.5, 2)),
        ((10, 0), (7.5, 2)),
        ((0, -10), (2, 0.5)),
        ((0, 10), (2, 5.5)),
        ((10, 10), (7.5, 5.5)),
    ],
    ids=("left", "right", "bottom", "top", "corner"),
)
def test_movement_respects_body_radius_at_every_map_edge(
    delta: tuple[float, float], expected: tuple[float, float]
) -> None:
    args = _scene([(2, 2)], [delta], width=8, height=6)
    result = _move(*args, collision_projection_passes=1, movement_substeps=1)
    wanted = np.asarray(args[0]).copy()
    wanted[0] = expected
    _assert_positions(result, wanted)


@pytest.mark.parametrize(
    ("kind", "angle", "corner", "direction"),
    [
        (OBSTACLE_TYPE_PILLAR, 0, False, (1, 0)),
        (OBSTACLE_TYPE_PILLAR, 0, False, (-1, 0)),
        (OBSTACLE_TYPE_PILLAR, 0, False, (0, 1)),
        (OBSTACLE_TYPE_PILLAR, 0, False, (0, -1)),
        (OBSTACLE_TYPE_PILLAR, 0, False, (0.6, 0.8)),
        (OBSTACLE_TYPE_WALL, 0, False, (-1, 0)),
        (OBSTACLE_TYPE_WALL, 0, False, (0, 1)),
        (OBSTACLE_TYPE_WALL, np.pi / 4, False, (-1, 0)),
        (OBSTACLE_TYPE_WALL, np.pi / 6, False, (0, -1)),
        (OBSTACLE_TYPE_WALL, 0, True, (0.6, 0.8)),
        (OBSTACLE_TYPE_WALL, np.pi / 4, True, (0.6, 0.8)),
    ],
    ids=(
        "pillar-right",
        "pillar-left",
        "pillar-top",
        "pillar-bottom",
        "pillar-diagonal",
        "wall-left-face",
        "wall-top-face",
        "rotated-wall-left-face",
        "rotated-wall-bottom-face",
        "wall-rounded-corner",
        "rotated-wall-rounded-corner",
    ),
)
def test_movement_stops_at_actual_pillar_or_rectangle_surface(
    kind: int, angle: float, corner: bool, direction: tuple[float, float]
) -> None:
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    normal = np.asarray(direction)
    contact = normal * 1.5
    start = normal * 3
    target = normal * 1.25
    if corner:
        contact = np.ones(2) + normal * 0.5
        start = np.ones(2) + normal * 2
        target = np.ones(2) + normal * 0.25
    center = np.array([10, 10])
    position = center + rotation @ start
    endpoint = center + rotation @ target
    obstacle = (kind, 10, 10, 1, 2, 2, angle, 1)
    args = _scene([position.tolist()], [(endpoint - position).tolist()], [obstacle])
    result = _move(*args, collision_projection_passes=1, movement_substeps=1)
    expected = np.asarray(args[0]).copy()
    expected[0] = center + rotation @ contact
    _assert_positions(result, expected)


@pytest.mark.parametrize(
    ("kind", "active"),
    [(OBSTACLE_TYPE_NONE, 1), (OBSTACLE_TYPE_PILLAR, 0), (OBSTACLE_TYPE_WALL, 0)],
    ids=("active-none", "inactive-pillar", "inactive-wall"),
)
def test_movement_ignores_inactive_and_none_obstacle_rows(
    kind: int, active: int
) -> None:
    args = _scene([(5, 5)], [(1, 0)], [(kind, 5, 5, 10, 10, 10, 0, active)])
    result = _move(*args, collision_projection_passes=1, movement_substeps=1)
    np.testing.assert_array_equal(result, args[0] + args[2])


@pytest.mark.parametrize("kind", [OBSTACLE_TYPE_PILLAR, OBSTACLE_TYPE_WALL])
@pytest.mark.parametrize("direction", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_charge_at_obstacle_center_uses_its_physical_approach(
    kind: int, direction: tuple[float, float]
) -> None:
    normal = np.asarray(direction, np.float32)
    position = np.array([10, 10], np.float32) + normal * 3
    args = _scene(
        [position.tolist()],
        [(-normal * 3).tolist()],
        [(kind, 10, 10, 1, 2, 2, 0, 1)],
    )
    result = _charge(*args, collision_projection_passes=4)
    expected = np.asarray(args[0]).copy()
    expected[0] = np.array([10, 10]) + normal * 1.5
    _assert_positions(result, expected)


@pytest.mark.parametrize(
    "permutation",
    [
        tuple(reversed(range(MAX_AGENT_SLOTS))),
        (5, 6, 7, 8, 9, 0, 1, 2, 3, 4),
        (3, 8, 0, 6, 1, 9, 4, 2, 7, 5),
    ],
    ids=("reverse", "team-blocks", "shuffle"),
)
def test_saved_body_contact_is_exact_after_agent_reordering(
    permutation: tuple[int, ...],
) -> None:
    args = _scene(
        [
            (7.7620930671691895, 4.916223049163818),
            (9.111735343933105, 5.424978256225586),
            (7.197245121002197, 5.7298688888549805),
            (9.499996185302734, 6.080760955810547),
            (4.828427791595459, 5.17156982421875),
            (12.237899780273438, 4.916223526000977),
            (10.888264656066895, 5.424979209899902),
            (12.802760124206543, 5.729866981506348),
            (10.499996185302734, 6.080760955810547),
            (15.17156982421875, 5.17156982421875),
        ]
    )
    order = np.asarray(permutation)
    mapped = tuple(
        value[order] if index in (0, 1, 2, 3, 4, 8, 9) else value
        for index, value in enumerate(args)
    )
    result = _move(*args, collision_projection_passes=1, movement_substeps=1)
    counterpart = _move(*mapped, collision_projection_passes=1, movement_substeps=1)
    assert np.isfinite(np.asarray(result)).all()
    assert not np.array_equal(result, args[0])
    _assert_same_position_bits(counterpart[np.argsort(order)], result)


def test_simultaneous_obstacle_contacts_are_exact_after_row_reordering() -> None:
    args = _scene(
        [(8, 10)],
        [(2, 0)],
        [
            (OBSTACLE_TYPE_PILLAR, 10, 9, 0.75, 0, 0, 0, 1),
            (OBSTACLE_TYPE_PILLAR, 10, 11, 0.75, 0, 0, 0, 1),
            (OBSTACLE_TYPE_WALL, 15, 15, 0, 2, 1, np.pi / 6, 1),
        ],
    )
    result = _move(*args, collision_projection_passes=4, movement_substeps=1)
    for order in (
        np.arange(MAX_OBSTACLE_SLOTS)[::-1],
        np.roll(np.arange(MAX_OBSTACLE_SLOTS), 13),
    ):
        mapped = (*args[:7], args[7][order], *args[8:])
        counterpart = _move(*mapped, collision_projection_passes=4, movement_substeps=1)
        _assert_same_position_bits(counterpart, result)
    actual = np.asarray(result)[0]
    assert np.isfinite(np.asarray(result)).all()
    assert 8 < actual[0] < 10
    assert actual[1] == 10
    for center in ((10, 9), (10, 11)):
        assert (
            np.linalg.norm(actual.astype(np.float64) - center)
            >= 1.25 - GEOMETRY_TOLERANCE
        )


@pytest.mark.parametrize("direction", [(1, 0), (0, 1)])
def test_coincident_bodies_use_different_intended_movements(
    direction: tuple[float, float],
) -> None:
    normal = np.asarray(direction, np.float32)
    args = _scene(
        [(5, 5), (5, 5)], [(normal * 0.25).tolist(), (-normal * 0.25).tolist()]
    )
    result = _move(*args, collision_projection_passes=4, movement_substeps=1)
    expected = np.asarray(args[0]).copy()
    expected[0] += normal * 0.5
    expected[1] -= normal * 0.5
    _assert_positions(result, expected)


def test_indistinguishable_coincident_bodies_do_not_choose_a_direction() -> None:
    args = _scene([(5, 5), (5, 5)], [(0.25, -0.125), (0.25, -0.125)])
    result = _move(*args, collision_projection_passes=4, movement_substeps=1)
    np.testing.assert_array_equal(result, args[0] + args[2])
    np.testing.assert_array_equal(result[0], result[1])


@pytest.mark.parametrize(
    ("charge", "positions", "deltas", "expected"),
    [
        (False, [(5, 5), (7, 5)], [(2, 0), (-2, 0)], [(5.5, 5), (6.5, 5)]),
        (True, [(5, 5), (7, 5)], [(1, 0), (-1, 0)], [(5.5, 5), (6.5, 5)]),
        (True, [(2, 6), (7, 6), (4, 6)], [(4, 0)], [(6, 6), (7, 6), (4, 6)]),
        (True, [(2, 6), (7, 6)], [(4, 0), (-4, 0)], [(6, 6), (3, 6)]),
        (False, [(2, 6), (7, 6)], [(4, 0), (-4, 0)], [(4, 6), (5, 6)]),
    ],
    ids=(
        "ordinary-crossing-keeps-incoming-sides",
        "charge-same-arrival-shares-contact",
        "charge-passes-midpath-body",
        "charge-exchanges-sides",
        "ordinary-exchange-is-blocked",
    ),
)
def test_ordinary_and_charge_body_contacts(
    charge: bool,
    positions: list[tuple[float, float]],
    deltas: list[tuple[float, float]],
    expected: list[tuple[float, float]],
) -> None:
    args = _scene(positions, deltas)
    if charge:
        result = _charge(*args, collision_projection_passes=4)
    else:
        result = _move(*args, collision_projection_passes=4, movement_substeps=1)
    wanted = np.asarray(args[0]).copy()
    wanted[: len(expected)] = expected
    _assert_positions(result, wanted)


def test_near_touching_ordinary_bodies_cannot_exchange_sides() -> None:
    left = float(np.nextafter(np.float32(0.5), np.float32(1)))
    args = _scene([(left, 5), (1.5, 5)], [(2, 0), (-2, 0)])
    assert np.float32(1.5) - np.float32(left) == np.nextafter(
        np.float32(1), np.float32(0)
    )
    result = _move(*args, collision_projection_passes=4, movement_substeps=4)
    _assert_positions(result, args[0])
    assert float(result[0, 0]) < float(result[1, 0])
    assert float(result[1, 0] - result[0, 0]) >= 1 - GEOMETRY_TOLERANCE


@pytest.mark.parametrize("joining", [False, True], ids=("blocking", "newly-joining"))
def test_only_newly_joining_overlap_can_recover_on_its_arrival_side(
    joining: bool,
) -> None:
    args = _scene([(5, 5), (5.5, 5)], [(1, 0)])
    if joining:
        args = (*args[:8], args[8].at[0].set(False), args[9])
    result = _move(*args, collision_projection_passes=4, movement_substeps=1)
    expected = np.asarray(args[0]).copy()
    expected[:2] = [(6.25, 5), (5.25, 5)] if joining else [(5.25, 5), (6.25, 5)]
    _assert_positions(result, expected)


def test_public_movement_separates_a_small_overlapping_chain() -> None:
    args = _scene([(9.6, 10), (10, 10), (10.4, 10)], radius=0.3)
    result = _move(*args, movement_substeps=1)
    positions = np.asarray(result)
    assert np.isfinite(positions).all()
    np.testing.assert_allclose(positions[:3, 1], 10, rtol=0, atol=GEOMETRY_TOLERANCE)
    assert np.all(np.diff(positions[:3, 0]) >= 0.6 - GEOMETRY_TOLERANCE)


def test_zero_body_sweeps_leave_body_contacts_but_keep_map_bounds() -> None:
    args = _scene([(0.5, 10), (2, 10)], [(-1, 0), (-1, 0)])
    result = _move(
        *args,
        agent_agent_overlap_projection_passes=0,
        collision_projection_passes=4,
        movement_substeps=1,
    )
    expected = np.asarray(args[0]).copy()
    expected[:2] = [(0.5, 10), (1, 10)]
    _assert_positions(result, expected)


def test_default_collision_budget_matches_28_literal_rounds() -> None:
    args = _scene([(0.5, 10), (2, 10)], [(0, 0), (-1, 0)])
    default = _move(*args, movement_substeps=1)
    explicit = _move(*args, collision_projection_passes=28, movement_substeps=1)
    short = _move(*args, collision_projection_passes=4, movement_substeps=1)
    np.testing.assert_array_equal(default, explicit)
    assert not np.array_equal(short, default)
    assert float(default[1, 0] - default[0, 0]) >= 1 - GEOMETRY_TOLERANCE


def test_saved_body_push_keeps_useful_wall_slide_and_static_clearance() -> None:
    args = _scene(
        [
            (3.08526349067688, 3.9732162952423096),
            (5.79088830947876, 0.5),
            (2.589773178100586, 1.2071067094802856),
            (6.79088830947876, 0.5),
            (1.5, 3.204050302505493),
            (5.468255043029785, 0.5),
            (4.271834850311279, 4.12699031829834),
            (9.1625394821167, 0.5),
            (12.984428405761719, 5.310550689697266),
            (7.79088830947876, 0.5),
        ],
        [
            (0.0, 0.0),
            (1.5, 0.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        [
            (1.0, 10.0, 7.5, 0.5, 0.0, 0.0, 0.0, 1.0),
            (1.0, 10.0, 5.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            (1.0, 10.0, 2.5, 0.5, 0.0, 0.0, 0.0, 1.0),
            (2.0, 7.5, 7.5, 0.0, 1.0, 3.0, 0.0, 1.0),
            (2.0, 12.5, 7.5, 0.0, 1.0, 3.0, 0.0, 1.0),
            (2.0, 7.5, 2.5, 0.0, 1.0, 3.0, 0.0, 1.0),
            (2.0, 12.5, 2.5, 0.0, 1.0, 3.0, 0.0, 1.0),
            (2.0, 4.0, 8.0, 0.0, 3.0, 0.5, 0.7853981852531433, 1.0),
            (2.0, 16.0, 8.0, 0.0, 3.0, 0.5, -0.7853981852531433, 1.0),
            (2.0, 16.0, 2.0, 0.0, 3.0, 0.5, 0.7853981852531433, 1.0),
            (2.0, 4.0, 2.0, 0.0, 3.0, 0.5, -0.7853981852531433, 1.0),
        ],
        height=10,
    )
    participants = (
        jnp.zeros(MAX_AGENT_SLOTS, dtype=bool).at[jnp.array([1, 3])].set(True)
    )
    args = (
        *args[:3],
        participants,
        participants,
        *args[5:8],
        participants,
        participants,
    )
    result = _move(*args)
    expected = np.asarray(args[0]).copy()
    expected[[1, 3]] = [[7.040887832641602, 0.5], [8.040887832641602, 0.5]]
    _assert_positions(result, expected)
