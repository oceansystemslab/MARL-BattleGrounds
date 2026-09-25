"""Hold the shared Red Zone scene-record cases for Python and browser tests.

RED_ZONE_CONTRACT_CASES is the one table of map-record rows for the numeric
contract of AuthorizedMapV2: each row names a raw map width, a candidate
red_zone record and whether the record must be accepted. The Python test checks
every row through the strict wire adapter; the browser fixture exporter copies
the same rows into the fixture so the browser normalizer checks them too.
Accepted rows hold only the two scoring strips at the float32 depth (left
(0, depth), right (float32(w32 - depth), w32)), including collapsed right
ranges; refused rows cover edge strips of the wrong size, one wrong range,
depths that are not float32 values or are subnormal, zero, negative or above
the float32 width, missing, extra and coerced fields, and a reversed range.

RED_ZONE_GEOMETRY_CASES lists recorded games (width, height, depth and each
team's pad x, with every pad inside the map) used to check that Oracle and
Agent POV scenes record the same strips as Core's side rule: depths 0, 5, 6,
5.5 with swapped banks, 0.1 on width 17.3, overlapping strips, a full-width
strip, both banks on the left, a 12 x 12 map, and the edge depths float32(1e-8)
on width 20 (right strip collapsed at (20, 20)), float32(12.1) on width 12.1
(both strips reach just past the raw width) and float32(1e-8) on width 12.1
(right strip collapsed at the float32 width). red_zone_env_config builds each
game's Team Deathmatch config, and expected_red_zone derives the record from
Core's spawn_bank_on_right and models.red_zone_x_range.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.axis_mappings import spawn_bank_on_right
from marl_battlegrounds.core.types import TASK_MODE_TDM, EnvConfig
from marl_battlegrounds.evaluation.models import float32_value, red_zone_x_range
from marl_battlegrounds.rendering.authorized_presentation import AuthorizedRedZoneV1

_WIDTH_12_1_AS_FLOAT32 = float32_value(12.1)
_TINY_DEPTH = float32_value(1e-8)
_LARGEST_SUBNORMAL = 2.0**-126 - 2.0**-149
_ABOVE_WIDTH_12_1 = float32_value(12.100002)

RED_ZONE_CONTRACT_CASES: tuple[dict[str, object], ...] = (
    {
        "label": "Left and right strips at width 20 and depth 5",
        "width": 20.0,
        "accepted": True,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [0.0, 5.0],
            "team_b_x_range": [15.0, 20.0],
        },
    },
    {
        "label": "Swapped banks at width 20 and depth 5",
        "width": 20.0,
        "accepted": True,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [15.0, 20.0],
            "team_b_x_range": [0.0, 5.0],
        },
    },
    {
        "label": "Both teams on the left",
        "width": 20.0,
        "accepted": True,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [0.0, 5.0],
            "team_b_x_range": [0.0, 5.0],
        },
    },
    {
        "label": "Tiny depth with a collapsed right range at width 20",
        "width": 20.0,
        "accepted": True,
        "red_zone": {
            "depth": _TINY_DEPTH,
            "team_a_x_range": [0.0, _TINY_DEPTH],
            "team_b_x_range": [20.0, 20.0],
        },
    },
    {
        "label": "Depth equal to the float32 width 12.1",
        "width": 12.1,
        "accepted": True,
        "red_zone": {
            "depth": _WIDTH_12_1_AS_FLOAT32,
            "team_a_x_range": [0.0, _WIDTH_12_1_AS_FLOAT32],
            "team_b_x_range": [0.0, _WIDTH_12_1_AS_FLOAT32],
        },
    },
    {
        "label": "Tiny depth with a right range collapsed at float32 width 12.1",
        "width": 12.1,
        "accepted": True,
        "red_zone": {
            "depth": _TINY_DEPTH,
            "team_a_x_range": [0.0, _TINY_DEPTH],
            "team_b_x_range": [_WIDTH_12_1_AS_FLOAT32, _WIDTH_12_1_AS_FLOAT32],
        },
    },
    {
        "label": "Edge strips of the wrong size",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [0.0, 1.0],
            "team_b_x_range": [19.0, 20.0],
        },
    },
    {
        "label": "One exact range beside one wrong range",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [0.0, 5.0],
            "team_b_x_range": [14.0, 20.0],
        },
    },
    {
        "label": "Depth that is not a float32 value",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.1,
            "team_a_x_range": [0.0, 5.1],
            "team_b_x_range": [14.9, 20.0],
        },
    },
    {
        "label": "Smallest subnormal depth",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 2.0**-149,
            "team_a_x_range": [0.0, 2.0**-149],
            "team_b_x_range": [20.0, 20.0],
        },
    },
    {
        "label": "Largest subnormal depth",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": _LARGEST_SUBNORMAL,
            "team_a_x_range": [0.0, _LARGEST_SUBNORMAL],
            "team_b_x_range": [20.0, 20.0],
        },
    },
    {
        "label": "Zero depth in a Red Zone record",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 0.0,
            "team_a_x_range": [0.0, 0.0],
            "team_b_x_range": [20.0, 20.0],
        },
    },
    {
        "label": "Negative depth",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": -5.0,
            "team_a_x_range": [0.0, -5.0],
            "team_b_x_range": [25.0, 20.0],
        },
    },
    {
        "label": "Depth above the float32 width",
        "width": 12.1,
        "accepted": False,
        "red_zone": {
            "depth": _ABOVE_WIDTH_12_1,
            "team_a_x_range": [0.0, _ABOVE_WIDTH_12_1],
            "team_b_x_range": [0.0, _WIDTH_12_1_AS_FLOAT32],
        },
    },
    {
        "label": "Missing Team B range",
        "width": 20.0,
        "accepted": False,
        "red_zone": {"depth": 5.0, "team_a_x_range": [0.0, 5.0]},
    },
    {
        "label": "Extra field",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [0.0, 5.0],
            "team_b_x_range": [15.0, 20.0],
            "points": 2,
        },
    },
    {
        "label": "Depth written as text",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": "5.0",
            "team_a_x_range": [0.0, 5.0],
            "team_b_x_range": [15.0, 20.0],
        },
    },
    {
        "label": "Range bound written as a boolean",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [False, 5.0],
            "team_b_x_range": [15.0, 20.0],
        },
    },
    {
        "label": "Reversed range",
        "width": 20.0,
        "accepted": False,
        "red_zone": {
            "depth": 5.0,
            "team_a_x_range": [5.0, 0.0],
            "team_b_x_range": [15.0, 20.0],
        },
    },
)


@dataclass(frozen=True, slots=True)
class RedZoneGeometryCase:
    label: str
    width: float
    height: float
    depth: float
    team_a_pad_x: float
    team_b_pad_x: float


RED_ZONE_GEOMETRY_CASES: tuple[RedZoneGeometryCase, ...] = (
    RedZoneGeometryCase("depth 0", 20.0, 12.0, 0.0, 1.5, 18.5),
    RedZoneGeometryCase("depth 5", 20.0, 12.0, 5.0, 1.5, 18.5),
    RedZoneGeometryCase("depth 6", 20.0, 12.0, 6.0, 1.5, 18.5),
    RedZoneGeometryCase("depth 5.5 swapped banks", 20.0, 12.0, 5.5, 18.5, 1.5),
    RedZoneGeometryCase("depth 0.1 on width 17.3", 17.3, 12.0, 0.1, 1.5, 15.8),
    RedZoneGeometryCase("overlapping strips", 20.0, 12.0, 12.0, 1.5, 18.5),
    RedZoneGeometryCase("full-width strips", 20.0, 12.0, 20.0, 1.5, 18.5),
    RedZoneGeometryCase("both banks on the left", 20.0, 12.0, 5.0, 1.5, 3.5),
    RedZoneGeometryCase("12 x 12 map", 12.0, 12.0, 5.0, 1.5, 10.5),
    RedZoneGeometryCase("depth 1e-8 on width 20", 20.0, 12.0, _TINY_DEPTH, 1.5, 18.5),
    RedZoneGeometryCase(
        "depth 12.1 on width 12.1", 12.1, 12.0, _WIDTH_12_1_AS_FLOAT32, 1.5, 10.6
    ),
    RedZoneGeometryCase("depth 1e-8 on width 12.1", 12.1, 12.0, _TINY_DEPTH, 1.5, 10.6),
)


def red_zone_env_config(case: RedZoneGeometryCase) -> EnvConfig:
    base = evaluation_env_config(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=5,
        team_deathmatch_red_zone_depth=case.depth,
        max_steps=1,
    )
    pads = (
        base.team_spawn_pad_positions.at[0, :, 0]
        .set(case.team_a_pad_x)
        .at[1, :, 0]
        .set(case.team_b_pad_x)
    )
    return base._replace(
        map_width=case.width,
        map_height=case.height,
        team_spawn_pad_positions=pads,
    )


def expected_red_zone(config: EnvConfig) -> AuthorizedRedZoneV1 | None:
    depth = float(config.team_deathmatch_red_zone_depth)
    if depth == 0.0:
        return None
    on_right = tuple(
        bool(value)
        for value in spawn_bank_on_right(
            jnp.asarray(config.team_spawn_pad_positions), config.map_width
        )
    )
    team_a, team_b = (
        red_zone_x_range(config.map_width, depth, side) for side in on_right
    )
    return AuthorizedRedZoneV1(
        depth=float32_value(depth),
        team_a_x_range=team_a,
        team_b_x_range=team_b,
    )
