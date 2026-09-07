"""Test-only scenario snapshots; never open the user's authoring store."""

from pathlib import Path

from scripts.dev.visual_debugger.authoring_compiler import (
    CompiledDevScenarioV1,
    compile_dev_scenario,
)
from scripts.dev.visual_debugger.authoring_models import DevScenarioDraftV1

SCENARIO_1_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_1_r34.json"
SCENARIO_1_SEMANTIC_DIGEST = (
    "d30c9fe5ac5fbe82e8831bdb2726cfbd3a2bf89a0749ac97ed492f79eb47f031"
)
SCENARIO_1_MAP_DIGEST = (
    "ac87824928b74f7db555b73a0127b56942f7747a6ee704c498924bf317149c63"
)
SCENARIO_1_CONFIG_DIGEST = (
    "f10fa90e736f09fd6a51484e3a44b53739f8e2b3bb1b5773a64899c41cc28f14"
)
SCENARIO_1_STATE_DIGEST = (
    "45039793469def83242f64dc01a68c68a7a3f030f917e2baae69a7557d29c20c"
)


def load_scenario_1_draft() -> DevScenarioDraftV1:
    """Read a fresh strict copy of the approved revision-34 test fixture."""
    return DevScenarioDraftV1.model_validate_json(
        SCENARIO_1_FIXTURE_PATH.read_text(encoding="utf-8"),
    )


def load_scenario_1() -> CompiledDevScenarioV1:
    """Compile the physical fixture through unchanged authoring authorities."""
    return compile_dev_scenario(load_scenario_1_draft())


SCENARIO_3_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_3_r9.json"
SCENARIO_3_SEMANTIC_DIGEST = (
    "03fafb3201c7cf9c18674a71d97b943b9e92162b9d445aed80e3b7724249f063"
)


def load_scenario_3_draft() -> DevScenarioDraftV1:
    """Read the approved r9 physical setup, not an official evaluation release."""
    return DevScenarioDraftV1.model_validate_json(
        SCENARIO_3_FIXTURE_PATH.read_text(encoding="utf-8"),
    )


def load_scenario_3() -> CompiledDevScenarioV1:
    """Compile Scenario 3 through the unchanged authoring authorities."""
    return compile_dev_scenario(load_scenario_3_draft())


SCENARIO_5_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_5_r9.json"
SCENARIO_5_SEMANTIC_DIGEST = (
    "20361030c886f508778305e79fd161b93595db407972a92bcd0a6d3d90760fd9"
)


def load_scenario_5_draft() -> DevScenarioDraftV1:
    """Read the approved r9 physical setup, not an official evaluation release."""
    return DevScenarioDraftV1.model_validate_json(
        SCENARIO_5_FIXTURE_PATH.read_text(encoding="utf-8"),
    )


def load_scenario_5() -> CompiledDevScenarioV1:
    """Compile Scenario 5 through the unchanged authoring authorities."""
    return compile_dev_scenario(load_scenario_5_draft())


# Scenario 6 r10 supplied the geometry and state. Its author-approved correction
# changes Team B's starting score from 18 to 19; this is not byte-identical r10.
SCENARIO_6_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "scenario_6_r10_score_17_19.json"
)
SCENARIO_6_SEMANTIC_DIGEST = (
    "6ad5b3ffa18de9defd77d8da7d3601cec7d8fceb1ea0e156f0b857c86d8331bb"
)


def load_scenario_6_draft() -> DevScenarioDraftV1:
    """Read the r10-derived 17-19 regression, not an official suite release."""
    return DevScenarioDraftV1.model_validate_json(
        SCENARIO_6_FIXTURE_PATH.read_text(encoding="utf-8"),
    )


def load_scenario_6() -> CompiledDevScenarioV1:
    """Compile Scenario 6 through the unchanged authoring authorities."""
    return compile_dev_scenario(load_scenario_6_draft())


SCENARIO_7_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scenario_7_r25.json"
SCENARIO_7_SEMANTIC_DIGEST = (
    "455117f81ff2b7dc8bb9f3d6abee294810c21e2af58fde61f237b20ec33e042e"
)


def load_scenario_7_draft() -> DevScenarioDraftV1:
    """Read the approved r25 physical setup, not an official evaluation release."""
    return DevScenarioDraftV1.model_validate_json(
        SCENARIO_7_FIXTURE_PATH.read_text(encoding="utf-8"),
    )


def load_scenario_7() -> CompiledDevScenarioV1:
    """Compile Scenario 7 through the unchanged authoring authorities."""
    return compile_dev_scenario(load_scenario_7_draft())
