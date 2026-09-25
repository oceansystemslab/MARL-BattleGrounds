"""Check that reading evaluation artifacts does not import simulator execution.

The frozen wire widths must match Core's current shapes. The context width is
the one exception with history: frames V1 and V2 keep 19 context columns
(CONTEXT_FEATURES_V1), while Core and frame V3 use 20 (CONTEXT_FEATURES_V2;
column 19 is the Red Zone depth). The host copies of the Team Deathmatch point
values (1 per death, 2 per Red Zone death) and of the largest score threshold
at depths 0.0 and 5.0 must equal Core's.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from marl_battlegrounds.core import config as core_config
from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.evaluation import models, wire_shapes

_WIRE_SHAPE_PARITY = (
    (wire_shapes.CONTEXT_FEATURES_V2, core_types.CONTEXT_FEATURES),
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
    assert all(
        wire_value == core_value for wire_value, core_value in _WIRE_SHAPE_PARITY
    )
    # Historical frames keep their recorded 19-column context layout.
    assert wire_shapes.CONTEXT_FEATURES_V1 == 19


_TEAM_DEATHMATCH_POINT_PARITY = (
    (
        models.TEAM_DEATHMATCH_POINTS_PER_DEATH,
        core_types.TEAM_DEATHMATCH_POINTS_PER_DEATH,
        1,
    ),
    (
        models.TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH,
        core_types.TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH,
        2,
    ),
    (
        models._maximum_team_deathmatch_score_threshold(0.0),  # pyright: ignore[reportPrivateUsage]
        core_config.maximum_team_deathmatch_score_threshold(0.0),
        16_777_212,
    ),
    (
        models._maximum_team_deathmatch_score_threshold(5.0),  # pyright: ignore[reportPrivateUsage]
        core_config.maximum_team_deathmatch_score_threshold(5.0),
        16_777_207,
    ),
)


def test_host_team_deathmatch_points_and_bound_match_core() -> None:
    # Rows: (host copy, Core owner, expected value).
    for host_value, core_value, expected in _TEAM_DEATHMATCH_POINT_PARITY:
        assert host_value == core_value == expected


@pytest.mark.parametrize(
    "module_name",
    (
        "marl_battlegrounds.evaluation.actor_projection",
        "marl_battlegrounds.evaluation.replay",
        "marl_battlegrounds.evaluation.replay_io",
        "marl_battlegrounds.evaluation.pov",
        "marl_battlegrounds.evaluation.scenario",
    ),
)
def test_replay_modules_import_without_array_or_capture_dependencies(
    module_name: str,
) -> None:
    script = f"""
import annotationlib
import importlib
import sys

module = importlib.import_module({module_name!r})

if {module_name!r} == "marl_battlegrounds.evaluation.actor_projection":
    reconstruction = module.reconstruct_shared_obs_sensor_source_bank_v1
    annotations = reconstruction.__annotations__
    if annotations.get("return") != "SharedObsSensorSourceBankV1":
        raise SystemExit("SharedObs reconstruction return annotation did not resolve")
    string_annotations = annotationlib.get_annotations(
        reconstruction,
        format=annotationlib.Format.STRING,
    )
    if string_annotations.get("return") != "SharedObsSensorSourceBankV1":
        raise SystemExit("SharedObs reconstruction annotation is not introspectable")

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


def test_v2_scenario_public_exports_are_host_only() -> None:
    export_names = (
        "build_scenario_evaluation_record_v2",
        "canonical_scenario_evaluation_record_json_bytes_v2",
        "load_scenario_evaluation_record_v2",
        "resolved_initial_state_digest_sha256_v2",
        "save_scenario_evaluation_record_v2",
        "validate_official_scenario_evaluation_record_v2",
        "validate_scenario_evaluation_record_v2",
    )
    script = f"""
import sys
from marl_battlegrounds import evaluation as evaluation_api

for export_name in {export_names!r}:
    if not callable(getattr(evaluation_api, export_name)):
        raise SystemExit("missing callable export: " + export_name)

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
