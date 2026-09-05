"""Test-only Scenario 1 snapshot; never opens the user's authoring store."""

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
