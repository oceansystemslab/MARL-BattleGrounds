"""Adversarial privacy, authority-join, and import-boundary proofs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from scripts.dev.visual_debugger.live_presentation import (
    build_live_no_shared_obs_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveNoSharedObsAuthorizedPresentationFrameV1,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    KeyboardCommandV1,
    RosterSelectionCommandV1,
)
from scripts.dev.visual_debugger.service import DebuggerService
from tests.test_visual_debugger_live_presentation import (
    _step_once,  # pyright: ignore[reportPrivateUsage]
    _switch_to_pov,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_visual_debugger_service import (
    _request,  # pyright: ignore[reportPrivateUsage]
    _service,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSliceV1,
    ActorPovCurrentSliceV1,
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return set(record) | {
            key for child in record.values() for key in _recursive_keys(child)
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {key for child in sequence for key in _recursive_keys(child)}
    return set()


def _recursive_string_values(value: object) -> set[str]:
    if type(value) is str:
        return {value}
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return {
            string
            for child in record.values()
            for string in _recursive_string_values(child)
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {
            string for child in sequence for string in _recursive_string_values(child)
        }
    return set()


def test_live_no_shared_excludes_oracle_ids_and_diagnostics() -> None:
    service = _service("basic_support")
    advanced = service.apply_command(
        _request(
            "adversarial-privacy-advance",
            base_revision=service.revision,
            command=KeyboardCommandV1(key="n"),
        )
    )
    assert advanced.outcome == "response"
    incoming = service.session.incoming_evaluation_view
    assert incoming is not None
    assert incoming.transition.events
    forbidden_values = {
        incoming.start_frame.frame_id,
        incoming.transition.transition_id,
        incoming.transition.start_frame_id,
        incoming.transition.successor_frame_id,
        incoming.successor_frame.frame_id,
        *(event.event_id for event in incoming.transition.events),
    }

    _switch_to_pov(service)
    result = service.current_presentation()
    assert result.outcome == "response"
    assert type(result.payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
    visual_events = result.payload.visual_events
    assert visual_events is not None
    assert visual_events.ordered_event_ids == tuple(
        event.event_id for event in visual_events.events
    )
    assert all(
        event_id.startswith(
            f"{visual_events.incoming_recipient_transition_id}:visual-event:"
        )
        for event_id in visual_events.ordered_event_ids
    )
    payload = result.payload.model_dump(mode="json")
    researcher_space = cast(dict[str, object], payload.pop("researcher_space"))
    corpse_overlay = cast(
        dict[str, object],
        payload.pop("local_oracle_corpse_overlay"),
    )
    assert corpse_overlay["overlay_kind"] == "local_oracle_corpse_overlay"
    assert corpse_overlay["source_frame_id"] == incoming.successor_frame.frame_id
    assert corpse_overlay["source_authority_epoch"] == (
        result.payload.source.source_authority_epoch
    )
    overlay_keys = _recursive_keys(corpse_overlay)
    overlay_strings = _recursive_string_values(corpse_overlay)
    keys = _recursive_keys(payload)
    strings = _recursive_string_values(payload)

    assert not (strings & forbidden_values)
    assert overlay_strings & forbidden_values == {incoming.successor_frame.frame_id}
    researcher_keys = _recursive_keys(researcher_space)
    assert not researcher_keys.intersection(
        {
            "actor_anchor",
            "center",
            "events",
            "latest_events",
            "map",
            "pending_route",
            "position",
            "ranges",
            "respawn_waves",
            "scene",
            "spawn_pads",
            "target_anchor",
            "visual_events",
        }
    )
    forbidden_exact_keys = {
        "aura_facts",
        "canonical_reward_by_agent",
        "canonical_reward_by_team",
        "combat_transition_facts",
        "death_facts",
        "owning_task_end_reason",
        "physical_facts",
        "regeneration_facts",
        "respawn_facts",
        "source_evidence",
        "source_frame_id",
        "spawn_shield_facts",
        "start_frame_id",
        "status_lifecycle_facts",
        "successor_frame_id",
        "transition_id",
    }
    assert not (overlay_keys & (forbidden_exact_keys - {"source_frame_id"}))
    for key in keys:
        assert key not in forbidden_exact_keys
        assert "global_slot" not in key
        assert "artifact" not in key
        assert "timeline" not in key
        assert "metric" not in key
        assert "completion" not in key
        assert "processing" not in key
        assert "canonical_" not in key
        assert "diagnostic" not in key
        assert "reward" not in key
        assert "terminated" not in key
        assert "truncated" not in key
        assert key not in {"cursor_generation", "choreography_generation"}
        if "digest" in key:
            assert key in {
                "authorized_endpoint_digest_sha256",
                "source_authorized_endpoint_digest_sha256",
            }
    for key in overlay_keys:
        assert "global_slot" not in key
        assert "artifact" not in key
        assert "timeline" not in key
        assert "metric" not in key
        assert "completion" not in key
        assert "processing" not in key
        assert "canonical_" not in key
        assert "diagnostic" not in key
        assert "reward" not in key
        assert "terminated" not in key
        assert "truncated" not in key
        if "digest" in key:
            assert key == "authorized_overlay_digest_sha256"


def _recipient_pair(
    service: DebuggerService,
) -> tuple[
    ActorPovCurrentSliceV1,
    ActorPovAdjacentTransitionSliceV1,
    ActorPovLiveDebuggerFrameV2,
]:
    session = service.session
    incoming = session.incoming_evaluation_view
    assert incoming is not None
    current_slice = build_actor_pov_current_slice_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        global_slot=session.controlled_global_slot,
        incoming_transition_view=incoming,
    )
    carrier = build_actor_pov_adjacent_transition_slice_v1(
        incoming,
        global_slot=session.controlled_global_slot,
    )
    raw = service.current_frame()
    assert type(raw) is ActorPovLiveDebuggerFrameV2
    return current_slice, carrier, raw


def test_live_no_shared_rejects_separately_valid_cross_swapped_pairs() -> None:
    service = _service()
    _step_once(service)
    _switch_to_pov(service)
    current_zero, carrier_zero, raw_zero = _recipient_pair(service)
    catalog = service.session.evaluation_context.static_mechanics_catalog
    context_zero = service.session.evaluation_context
    global_frame_zero = service.session.current_evaluation_frame
    previous_global_frame_zero = (
        None
        if service.session.incoming_evaluation_view is None
        else service.session.incoming_evaluation_view.start_frame
    )
    zero_presentation = service.current_presentation()
    assert zero_presentation.outcome == "response"
    assert (
        type(zero_presentation.payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
    )
    researcher_zero = zero_presentation.payload.researcher_space
    accepted_zero = build_live_no_shared_obs_authorized_presentation_v1(
        current_zero,
        carrier_zero,
        raw_zero,
        global_context=context_zero,
        current_global_frame=global_frame_zero,
        previous_global_frame=previous_global_frame_zero,
        public_catalog=catalog,
        incoming_visual_events=build_visual_event_batch_v2(
            cast(EvaluationTransitionViewV1, service.session.incoming_evaluation_view)
        ),
        researcher_space=researcher_zero,
    )

    switched = service.apply_command(
        _request(
            "adversarial-switch-recipient",
            base_revision=service.revision,
            command=RosterSelectionCommandV1(role="control", global_slot=1),
        )
    )
    assert switched.outcome == "response"
    current_one, carrier_one, raw_one = _recipient_pair(service)
    context_one = service.session.evaluation_context
    global_frame_one = service.session.current_evaluation_frame
    previous_global_frame_one = (
        None
        if service.session.incoming_evaluation_view is None
        else service.session.incoming_evaluation_view.start_frame
    )
    one_presentation = service.current_presentation()
    assert one_presentation.outcome == "response"
    assert (
        type(one_presentation.payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
    )
    researcher_one = one_presentation.payload.researcher_space
    accepted_one = build_live_no_shared_obs_authorized_presentation_v1(
        current_one,
        carrier_one,
        raw_one,
        global_context=context_one,
        current_global_frame=global_frame_one,
        previous_global_frame=previous_global_frame_one,
        public_catalog=catalog,
        incoming_visual_events=build_visual_event_batch_v2(
            cast(EvaluationTransitionViewV1, service.session.incoming_evaluation_view)
        ),
        researcher_space=researcher_one,
    )
    assert accepted_zero.source.source_recipient_public_agent_id != (
        accepted_one.source.source_recipient_public_agent_id
    )

    with pytest.raises(
        ValueError,
        match="carrier does not enter the current slice",
    ):
        build_live_no_shared_obs_authorized_presentation_v1(
            current_zero,
            carrier_one,
            raw_zero,
            global_context=context_zero,
            current_global_frame=global_frame_zero,
            previous_global_frame=previous_global_frame_zero,
            public_catalog=catalog,
            incoming_visual_events=build_visual_event_batch_v2(
                cast(
                    EvaluationTransitionViewV1,
                    service.session.incoming_evaluation_view,
                )
            ),
            researcher_space=researcher_zero,
        )
    with pytest.raises(
        ValueError,
        match="carrier does not enter the current slice",
    ):
        build_live_no_shared_obs_authorized_presentation_v1(
            current_one,
            carrier_zero,
            raw_one,
            global_context=context_one,
            current_global_frame=global_frame_one,
            previous_global_frame=previous_global_frame_one,
            public_catalog=catalog,
            incoming_visual_events=build_visual_event_batch_v2(
                cast(
                    EvaluationTransitionViewV1,
                    service.session.incoming_evaluation_view,
                )
            ),
            researcher_space=researcher_one,
        )


_ISOLATED_IMPORT_PROBE = """
import importlib
import json
import sys

importlib.import_module(sys.argv[1])
forbidden = {
    name
    for name in sys.modules
    if name in {
        "jax",
        "jaxlib",
        "numpy",
        "scripts.dev.visual_debugger.presentation",
        "scripts.dev.visual_debugger.replay_protocol",
        "scripts.dev.visual_debugger.replay_service",
    }
    or name.startswith(("jax.", "jaxlib.", "numpy."))
}
print(json.dumps(sorted(forbidden)))
"""


def test_live_presentation_import_isolated_from_replay_and_array_runtimes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _ISOLATED_IMPORT_PROBE,
            "scripts.dev.visual_debugger.live_presentation",
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_replay_service_import_does_not_load_live_presentation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, json, sys; "
                "importlib.import_module("
                "'scripts.dev.visual_debugger.replay_service'); "
                "print(json.dumps("
                "'scripts.dev.visual_debugger.live_presentation' in sys.modules))"
            ),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) is False
