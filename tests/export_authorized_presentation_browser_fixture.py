"""Generate exact Python-owned browser fixtures for CP2.7 tests only."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.test_rendering_authorized_inspection import (
    _InspectionCases,
    inspection_cases,
)
from tests.test_visual_debugger_presentation_protocol_v1 import (
    _FiveFrames,
    _live_no_shared_at,
    _live_oracle_at,
    _replay_no_shared_at,
    _replay_oracle_at,
    _replay_shared_at,
    five_frames,
)
from tests.test_visual_debugger_replay_service import (
    _presentation_service,
    _ServiceCases,
    service_cases,
)
from tests.test_visual_debugger_service import _service
from tests.visual_debugger_fixtures import debugger_test_launch_specification


class _WrappedFixture0[T](Protocol):
    __wrapped__: Callable[[], T]


class _WrappedFixture1[T, R](Protocol):
    __wrapped__: Callable[[T], R]


def _live_pov_service() -> DebuggerService:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    return DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="cp2-7-live-pov-fixture",
    )


def _pair(service: DebuggerService | ReplayViewerService) -> dict[str, object]:
    raw = service.current_frame()
    result = service.current_presentation()
    if result.outcome != "response":
        raise RuntimeError("fixture service did not produce a presentation response")
    pair: dict[str, object] = {
        "transport": raw.model_dump(mode="json"),
        "presentation": result.payload.model_dump(mode="json"),
    }
    if isinstance(service, ReplayViewerService):
        pair["timeline"] = service.current_timeline().model_dump(mode="json")
    return pair


def render_fixture() -> str:
    cases = cast("_WrappedFixture0[_InspectionCases]", inspection_cases).__wrapped__()
    frames = cast(
        "_WrappedFixture1[_InspectionCases, _FiveFrames]", five_frames
    ).__wrapped__(cases)
    replay_cases = cast("_WrappedFixture0[_ServiceCases]", service_cases).__wrapped__()
    pair_services = {
        "live_oracle": _service(),
        "live_no_shared_obs_agent_pov": _live_pov_service(),
        "replay_oracle": _presentation_service(
            replay_cases,
            "oracle",
            viewer_session_id="cp2-7-replay-oracle-fixture",
        ),
        "replay_no_shared_obs_agent_pov": _presentation_service(
            replay_cases,
            "no_shared_obs",
            viewer_session_id="cp2-7-replay-no-shared-fixture",
        ),
        "replay_shared_obs_agent_pov": _presentation_service(
            replay_cases,
            "shared_obs",
            viewer_session_id="cp2-7-replay-shared-fixture",
        ),
    }
    pairs = {kind: _pair(service) for kind, service in pair_services.items()}
    continuity_session = "cp2-7-replay-audience-switch-fixture"
    continuity_pairs = {
        "oracle": _pair(
            ReplayViewerService(
                replay_cases.shared.bundle,
                initial_frame_index=1,
                view_mode="researcher",
                selected_global_slot=0,
                viewer_session_id=continuity_session,
            )
        ),
        "shared_obs": _pair(
            ReplayViewerService(
                replay_cases.shared.bundle,
                initial_frame_index=1,
                view_mode="pov",
                pov_global_slot=0,
                viewer_session_id=continuity_session,
            )
        ),
    }
    final_index = len(cases.no_shared.transitions)
    shared_final_index = len(cases.shared.transitions)
    state_cases = {
        "live_oracle_frame_zero": _live_oracle_at(
            cases,
            frame_index=0,
            session="cp2-7-live-oracle-zero",
        ),
        "live_no_shared_frame_zero": _live_no_shared_at(
            cases,
            frame_index=0,
            session="cp2-7-live-no-shared-zero",
        ),
        "replay_oracle_frame_zero": _replay_oracle_at(
            cases,
            frame_index=0,
            selected_internal_slot=0,
            session="cp2-7-replay-oracle-zero",
        ),
        "replay_oracle_final_selected": _replay_oracle_at(
            cases,
            frame_index=final_index,
            selected_internal_slot=0,
            session="cp2-7-replay-oracle-final-selected",
        ),
        "replay_oracle_final_unselected": _replay_oracle_at(
            cases,
            frame_index=final_index,
            selected_internal_slot=None,
            session="cp2-7-replay-oracle-final-unselected",
        ),
        "replay_no_shared_frame_zero": _replay_no_shared_at(
            cases,
            frame_index=0,
            session="cp2-7-replay-no-shared-zero",
        ),
        "replay_no_shared_final": _replay_no_shared_at(
            cases,
            frame_index=final_index,
            session="cp2-7-replay-no-shared-final",
        ),
        "replay_shared_frame_zero": _replay_shared_at(
            cases,
            frame_index=0,
            session="browser-replay-shared-source-material-metric-processing",
        ),
        "replay_shared_final": _replay_shared_at(
            cases,
            frame_index=shared_final_index,
            session="cp2-7-replay-shared-final",
        ),
    }
    payload = {
        "schema_version": 1,
        "presentations": {
            frame.presentation_kind: frame.model_dump(mode="json")
            for frame in frames.rows
        },
        "pairs": pairs,
        "continuity_pairs": continuity_pairs,
        "state_cases": {
            name: frame.model_dump(mode="json") for name, frame in state_cases.items()
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    destination = (
        repository_root
        / "web"
        / "visual_debugger"
        / "tests"
        / "fixtures"
        / "authorized-presentations-v1.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_fixture(), encoding="utf-8")


if __name__ == "__main__":
    main()
