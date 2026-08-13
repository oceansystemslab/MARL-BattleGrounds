"""Import-boundary proof for host-only evaluation artifact modules."""

from __future__ import annotations

import subprocess
import sys

import pytest

from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.evaluation import wire_shapes

_WIRE_SHAPE_PARITY = (
    (wire_shapes.CONTEXT_FEATURES_V1, core_types.CONTEXT_FEATURES),
    (wire_shapes.ENVIRONMENT_DIMENSIONS_V1, core_types.ENVIRONMENT_DIMENSIONS),
    (wire_shapes.MAX_AGENT_SLOTS_V1, core_types.MAX_AGENT_SLOTS),
    (wire_shapes.MAX_AGENTS_PER_TEAM_V1, core_types.MAX_AGENTS_PER_TEAM),
    (wire_shapes.MAX_OBJECTIVE_SLOTS_V1, core_types.MAX_OBJECTIVE_SLOTS),
    (wire_shapes.MAX_OBSTACLE_SLOTS_V1, core_types.MAX_OBSTACLE_SLOTS),
    (wire_shapes.NUM_CLASSES_V1, core_types.NUM_CLASSES),
    (wire_shapes.NUM_MOVE_ACTIONS_V1, core_types.NUM_MOVE_ACTIONS),
    (wire_shapes.NUM_SLOW_CHANNELS_V1, core_types.NUM_SLOW_CHANNELS),
    (wire_shapes.NUM_STUN_CHANNELS_V1, core_types.NUM_STUN_CHANNELS),
    (wire_shapes.NUM_TARGET_ACTIONS_V1, core_types.NUM_TARGET_ACTIONS),
    (wire_shapes.NUM_TEAMS_V1, core_types.NUM_TEAMS),
    (wire_shapes.NUM_ULTIMATE_ACTIONS_V1, core_types.NUM_ULTIMATE_ACTIONS),
    (wire_shapes.OBJECTIVE_FEATURES_V1, core_types.OBJECTIVE_FEATURES),
    (wire_shapes.OBSTACLE_FEATURES_V1, core_types.OBSTACLE_FEATURES),
    (wire_shapes.SELF_FEATURES_V1, core_types.SELF_FEATURES),
    (wire_shapes.UNIT_FEATURES_V1, core_types.UNIT_FEATURES),
)

_FORBIDDEN_HOST_IMPORT_PREFIXES = ("jax", "jaxlib", "numpy")
_FORBIDDEN_DIRECT_MODULES = (
    "marl_battlegrounds.core.types",
    "marl_battlegrounds.evaluation.capture",
    "marl_battlegrounds.evaluation.catalog",
)


def test_evaluation_v1_wire_shapes_match_current_core_contract() -> None:
    """V1 stays explicit while current capture and wire dimensions agree exactly."""
    assert all(
        wire_value == core_value for wire_value, core_value in _WIRE_SHAPE_PARITY
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "marl_battlegrounds.evaluation.replay",
        "marl_battlegrounds.evaluation.replay_io",
        "marl_battlegrounds.evaluation.pov",
    ),
)
def test_replay_modules_import_without_array_or_capture_dependencies(
    module_name: str,
) -> None:
    """A fresh artifact reader must not initialize simulator capture dependencies."""
    script = f"""
import importlib
import sys

importlib.import_module({module_name!r})

for loaded_name in sys.modules:
    if loaded_name in {_FORBIDDEN_DIRECT_MODULES!r}:
        raise SystemExit("unexpected host dependency: " + loaded_name)
    if any(
        loaded_name == prefix or loaded_name.startswith(prefix + ".")
        for prefix in {_FORBIDDEN_HOST_IMPORT_PREFIXES!r}
    ):
        raise SystemExit("unexpected array-runtime import: " + loaded_name)
"""

    result = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
