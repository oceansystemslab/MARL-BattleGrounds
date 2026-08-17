"""Focused CP2.2 SharedObs visual-union authority and privacy proofs."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
)

from marl_battlegrounds.evaluation.models import (
    StaticMechanicsCatalogV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovAxisMappingV1,
    ActorPovSpawnLifecycleV1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.rendering import evaluation_adapter as adapter
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    SharedObsAgentObservationProvenanceV1,
    SharedObsAuthorizedScenePartsV1,
    SharedObsAuthorizedSensorSourceV1,
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedClassDocumentationProfileAvailableV1,
    AuthorizedClassDocumentationProfileUnavailableV1,
    AuthorizedClassMechanicsV2,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
    build_shared_obs_source_material_projection_v1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_ALIVE_V1,
    AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1,
    AGENT_FEATURE_CURRENT_HEALTH_V1,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1,
    AGENT_FEATURE_X_V1,
    CONTEXT_FEATURE_MAP_WIDTH_V1,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _SharedCase:
    trajectory: CapturedEvaluationTrajectory
    projection_by_slot: dict[int, SharedObsSourceMaterialProjectionV1]

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self.projection_by_slot))


@pytest.fixture(scope="module")
def shared_case() -> _SharedCase:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        execution_information_mode="shared_obs",
    )
    projection_by_slot = {
        roster.global_slot: build_shared_obs_source_material_projection_v1(
            trajectory.context,
            trajectory.frames[0],
            selected_global_slot=roster.global_slot,
        )
        for roster in trajectory.context.roster
        if roster.configured_active
    }
    return _SharedCase(
        trajectory=trajectory,
        projection_by_slot=projection_by_slot,
    )


def _replace_tuple_item[T](
    values: tuple[T, ...], index: int, value: T
) -> tuple[T, ...]:
    rows = list(values)
    rows[index] = value
    return tuple(rows)


def _changed_feature_row(
    row: tuple[float, ...],
    feature_index: int,
    value: float,
) -> tuple[float, ...]:
    return _replace_tuple_item(row, feature_index, value)


def _with_frame(
    projection: SharedObsSourceMaterialProjectionV1,
    **updates: object,
) -> SharedObsSourceMaterialProjectionV1:
    frame = replace(projection.base_sensor_frame, **updates)
    selected = projection.base_sensor_scene.self_actor
    scene = adapter._shared_obs_base_sensor_scene(  # pyright: ignore[reportPrivateUsage]
        frame,
        selected_global_slot=selected.global_slot,
        selected_team_local_slot=selected.team_local_slot,
        configured_team_id=selected.team_id,
        class_id=selected.class_id,
        axis_mapping=projection.axis_mapping,
    )
    return replace(
        projection,
        base_sensor_frame=frame,
        base_sensor_scene=scene,
    )


def _with_axis_mapping(
    projection: SharedObsSourceMaterialProjectionV1,
    axis_mapping: ActorPovAxisMappingV1,
) -> SharedObsSourceMaterialProjectionV1:
    selected = projection.base_sensor_scene.self_actor
    scene = adapter._shared_obs_base_sensor_scene(  # pyright: ignore[reportPrivateUsage]
        projection.base_sensor_frame,
        selected_global_slot=selected.global_slot,
        selected_team_local_slot=selected.team_local_slot,
        configured_team_id=selected.team_id,
        class_id=selected.class_id,
        axis_mapping=axis_mapping,
    )
    return replace(projection, axis_mapping=axis_mapping, base_sensor_scene=scene)


def _with_self_features(
    projection: SharedObsSourceMaterialProjectionV1,
    self_features: tuple[float, ...],
) -> SharedObsSourceMaterialProjectionV1:
    self_slot = projection.base_sensor_scene.self_actor.global_slot
    ally_row = projection.ally_observation_row_global_slot_by_id.index(self_slot)
    ally_rows = _replace_tuple_item(
        projection.base_sensor_frame.ally_unit_features,
        ally_row,
        self_features,
    )
    return _with_frame(
        projection,
        self_features=self_features,
        ally_unit_features=ally_rows,
    )


def _with_relation_row(
    projection: SharedObsSourceMaterialProjectionV1,
    *,
    global_slot: int,
    row: tuple[float, ...],
    visible: bool,
) -> SharedObsSourceMaterialProjectionV1:
    if global_slot in projection.ally_observation_row_global_slot_by_id:
        observation_row = projection.ally_observation_row_global_slot_by_id.index(
            global_slot
        )
        unit_rows = _replace_tuple_item(
            projection.base_sensor_frame.ally_unit_features,
            observation_row,
            row,
        )
        visibility = _replace_tuple_item(
            projection.base_sensor_frame.ally_visibility_mask,
            observation_row,
            visible,
        )
        return _with_frame(
            projection,
            ally_unit_features=unit_rows,
            ally_visibility_mask=visibility,
        )
    observation_row = projection.enemy_observation_row_global_slot_by_id.index(
        global_slot
    )
    unit_rows = _replace_tuple_item(
        projection.base_sensor_frame.enemy_unit_features,
        observation_row,
        row,
    )
    visibility = _replace_tuple_item(
        projection.base_sensor_frame.enemy_visibility_mask,
        observation_row,
        visible,
    )
    return _with_frame(
        projection,
        enemy_unit_features=unit_rows,
        enemy_visibility_mask=visibility,
    )


def _with_lifecycle(
    projection: SharedObsSourceMaterialProjectionV1,
    **updates: object,
) -> SharedObsSourceMaterialProjectionV1:
    payload = projection.base_sensor_frame.spawn_lifecycle.model_dump(mode="python")
    payload.update(updates)
    lifecycle = ActorPovSpawnLifecycleV1.model_validate(payload)
    return _with_frame(projection, spawn_lifecycle=lifecycle)


def _source_tuple(
    case: _SharedCase,
    recipient_slot: int,
    *,
    replacements: dict[int, SharedObsSourceMaterialProjectionV1] | None = None,
) -> tuple[SharedObsSourceMaterialProjectionV1, ...]:
    changed = replacements or {}
    return tuple(
        changed.get(slot, case.projection_by_slot[slot])
        for slot in case.active_slots
        if slot != recipient_slot
    )


def _build(
    case: _SharedCase,
    recipient_slot: int,
    *,
    recipient: SharedObsSourceMaterialProjectionV1 | None = None,
    replacements: dict[int, SharedObsSourceMaterialProjectionV1] | None = None,
    sources: tuple[SharedObsSourceMaterialProjectionV1, ...] | None = None,
    catalog: StaticMechanicsCatalogV1 | None = None,
    authority_session_id: str = "shared-authority",
) -> SharedObsAuthorizedScenePartsV1:
    return build_shared_obs_authorized_scene_v1(
        recipient or case.projection_by_slot[recipient_slot],
        all_active_nonrecipient_source_material=(
            sources
            if sources is not None
            else _source_tuple(
                case,
                recipient_slot,
                replacements=replacements,
            )
        ),
        public_catalog=catalog or case.trajectory.context.static_mechanics_catalog,
        authority_session_id=authority_session_id,
    )


def _canonical_bytes(value: object) -> bytes:
    return TypeAdapter(type(value)).dump_json(value)


def _catalog_from_payload(payload: dict[str, object]) -> StaticMechanicsCatalogV1:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return StaticMechanicsCatalogV1.model_validate(payload)


def test_false_diagonal_keeps_unconditional_recipient_self(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    self_row = recipient.base_sensor_scene.self_actor.team_local_slot
    changed = _with_frame(
        recipient,
        ally_visibility_mask=_replace_tuple_item(
            recipient.base_sensor_frame.ally_visibility_mask,
            self_row,
            False,
        ),
    )

    parts = _build(shared_case, 0, recipient=changed)

    assert sum(agent.relation == "self" for agent in parts.scene.agents) == 1
    assert parts.recipient_public_agent_id == "agent-slot-0"
    assert parts.scene.agents[0].public_agent_id == "agent-slot-0"


@pytest.mark.parametrize(
    ("recipient_slot", "expected_agents", "expected_sources"),
    (
        (0, ("agent-slot-0", "agent-slot-1", "agent-slot-2"), (0, 1, 2)),
        (5, ("agent-slot-5", "agent-slot-6"), (5, 6)),
    ),
)
def test_hand_enumerated_team_unions(
    shared_case: _SharedCase,
    recipient_slot: int,
    expected_agents: tuple[str, ...],
    expected_sources: tuple[int, ...],
) -> None:
    parts = _build(shared_case, recipient_slot)

    assert tuple(agent.public_agent_id for agent in parts.scene.agents) == (
        expected_agents
    )
    assert tuple(
        source.source_public_agent_id for source in parts.authorized_sensor_sources
    ) == tuple(f"agent-slot-{slot}" for slot in expected_sources)
    assert tuple(pad.team_id for pad in parts.scene.spawn_pads) == (1,) * 5 + (2,) * 5
    assert tuple(wave.team_id for wave in parts.scene.respawn_waves) == (1, 2)


def test_true_source_admission_adds_only_source_visible_rows(
    shared_case: _SharedCase,
) -> None:
    opponent_self = shared_case.projection_by_slot[5].base_sensor_frame.self_features
    source_one = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=opponent_self,
        visible=True,
    )

    parts = _build(shared_case, 0, replacements={1: source_one})

    assert tuple(agent.public_agent_id for agent in parts.scene.agents) == (
        "agent-slot-0",
        "agent-slot-1",
        "agent-slot-2",
        "agent-slot-5",
    )
    opponent_provenance = parts.agent_observation_provenance[-1]
    assert tuple(
        source.source_public_agent_id
        for source in opponent_provenance.observation_sources
    ) == ("agent-slot-1",)


def test_visible_self_diagonal_requires_exact_full_row_and_collapses(
    shared_case: _SharedCase,
) -> None:
    source = shared_case.projection_by_slot[1]
    self_row = source.base_sensor_scene.self_actor.team_local_slot
    assert source.base_sensor_frame.ally_visibility_mask[self_row]
    equal_parts = _build(shared_case, 0)
    assert (
        tuple(agent.public_agent_id for agent in equal_parts.scene.agents).count(
            "agent-slot-1"
        )
        == 1
    )

    conflicting_rows = _replace_tuple_item(
        source.base_sensor_frame.ally_unit_features,
        self_row,
        _changed_feature_row(
            source.base_sensor_frame.ally_unit_features[self_row],
            AGENT_FEATURE_X_V1,
            source.base_sensor_frame.ally_unit_features[self_row][AGENT_FEATURE_X_V1]
            + 0.25,
        ),
    )
    conflicting = _with_frame(source, ally_unit_features=conflicting_rows)
    with pytest.raises(ValueError, match="self diagonal conflicts"):
        _build(shared_case, 0, replacements={1: conflicting})


def test_availability_topology_rejects_invalid_cells_and_missing_rows(
    shared_case: _SharedCase,
) -> None:
    projection = shared_case.projection_by_slot[0]
    rows = projection.sensor_source_availability
    for index in (0, 3, 5):
        with pytest.raises(ValueError, match="available SharedObs sources"):
            replace(rows[index], recorded_available=True)
    with pytest.raises(ValueError, match="full source axis"):
        replace(projection, sensor_source_availability=rows[:-1])


def test_complete_nonrecipient_source_set_and_epoch_fail_closed(
    shared_case: _SharedCase,
) -> None:
    sources = _source_tuple(shared_case, 0)
    with pytest.raises(ValueError, match="every and only active nonrecipient"):
        _build(shared_case, 0, sources=sources[:-1])
    with pytest.raises(ValueError, match="duplicated"):
        _build(shared_case, 0, sources=(*sources[:-1], sources[0]))

    foreign_trajectory = captured_evaluation_trajectory(
        transition_count=0,
        execution_information_mode="shared_obs",
        episode_id="foreign-episode",
    )
    foreign = build_shared_obs_source_material_projection_v1(
        foreign_trajectory.context,
        foreign_trajectory.frames[0],
        selected_global_slot=1,
    )
    with pytest.raises(ValueError, match="recipient epoch"):
        _build(shared_case, 0, replacements={1: foreign})

    next_source = build_shared_obs_source_material_projection_v1(
        shared_case.trajectory.context,
        shared_case.trajectory.frames[1],
        selected_global_slot=1,
        transition_view=adapter.EvaluationTransitionViewV1(
            context=shared_case.trajectory.context,
            start_frame=shared_case.trajectory.frames[0],
            transition=shared_case.trajectory.transitions[0],
            successor_frame=shared_case.trajectory.frames[1],
        ),
    )
    with pytest.raises(ValueError, match="recipient epoch"):
        _build(shared_case, 0, replacements={1: next_source})

    forged_frame = copy.copy(shared_case.projection_by_slot[5].base_sensor_frame)
    object.__setattr__(forged_frame, "source_material_frame_id", "forged-local-id")
    forged_projection = copy.copy(shared_case.projection_by_slot[5])
    object.__setattr__(forged_projection, "base_sensor_frame", forged_frame)
    with pytest.raises(ValueError, match="base-frame declaration is not canonical"):
        _build(shared_case, 0, replacements={5: forged_projection})


def test_incoming_oracle_identity_is_inert_for_every_shared_source_kind(
    shared_case: _SharedCase,
) -> None:
    baseline = _build(shared_case, 0)

    def forged_incoming(
        projection: SharedObsSourceMaterialProjectionV1,
    ) -> SharedObsSourceMaterialProjectionV1:
        forged = copy.copy(projection)
        object.__setattr__(forged, "incoming_transition_id", "oracle-secret")
        return forged

    recipient_changed = _build(
        shared_case,
        0,
        recipient=forged_incoming(shared_case.projection_by_slot[0]),
    )
    admitted_changed = _build(
        shared_case,
        0,
        replacements={1: forged_incoming(shared_case.projection_by_slot[1])},
    )
    unavailable_changed = _build(
        shared_case,
        0,
        replacements={5: forged_incoming(shared_case.projection_by_slot[5])},
    )

    expected = _canonical_bytes(baseline)
    assert _canonical_bytes(recipient_changed) == expected
    assert _canonical_bytes(admitted_changed) == expected
    assert _canonical_bytes(unavailable_changed) == expected


def test_forged_recipient_scene_fact_must_rederive_from_used_frame(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    self_actor = recipient.base_sensor_scene.self_actor
    forged_self = replace(
        self_actor,
        position=(self_actor.position[0] + 0.25, self_actor.position[1]),
    )
    forged_scene = replace(recipient.base_sensor_scene, self_actor=forged_self)
    forged = copy.copy(recipient)
    object.__setattr__(forged, "base_sensor_scene", forged_scene)

    with pytest.raises(ValueError, match="scene must derive from its used frame"):
        _build(shared_case, 0, recipient=forged)


def test_each_recipient_scene_only_branch_rejects_but_matched_frame_accepts(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    scene = recipient.base_sensor_scene
    changed_map = replace(scene.map, width=scene.map.width + 1.0)
    changed_self = replace(
        scene.self_actor,
        position=(scene.self_actor.position[0] + 0.25, scene.self_actor.position[1]),
    )
    first_body = next(
        body
        for body in scene.visible_bodies
        if body.public_agent_id != scene.self_actor.public_agent_id
    )
    first_body_index = scene.visible_bodies.index(first_body)
    changed_body = replace(
        first_body,
        position=(first_body.position[0] + 0.25, first_body.position[1]),
    )
    changed_pad = replace(
        scene.spawn_pads[0],
        position=(
            scene.spawn_pads[0].position[0] + 0.25,
            scene.spawn_pads[0].position[1],
        ),
    )
    changed_wave = replace(
        scene.respawn_waves[0],
        countdown_steps=scene.respawn_waves[0].countdown_steps - 1,
    )
    scene_variants = (
        replace(scene, map=changed_map),
        replace(scene, self_actor=changed_self),
        replace(
            scene,
            visible_bodies=_replace_tuple_item(
                scene.visible_bodies,
                first_body_index,
                changed_body,
            ),
        ),
        replace(
            scene,
            spawn_pads=_replace_tuple_item(scene.spawn_pads, 0, changed_pad),
        ),
        replace(
            scene,
            respawn_waves=_replace_tuple_item(scene.respawn_waves, 0, changed_wave),
        ),
    )
    for variant in scene_variants:
        forged = copy.copy(recipient)
        object.__setattr__(forged, "base_sensor_scene", variant)
        with pytest.raises(ValueError, match="scene must derive from its used frame"):
            _build(shared_case, 0, recipient=forged)

    context = list(recipient.base_sensor_frame.context_features)
    context[CONTEXT_FEATURE_MAP_WIDTH_V1] += 1.0
    matched = _with_frame(recipient, context_features=tuple(context))
    accepted = _build(shared_case, 0, recipient=matched)
    assert accepted.scene.map.width == scene.map.width + 1.0

    changed_self_row = _changed_feature_row(
        recipient.base_sensor_frame.self_features,
        AGENT_FEATURE_X_V1,
        recipient.base_sensor_frame.self_features[AGENT_FEATURE_X_V1] + 0.25,
    )
    matched_self = _with_self_features(recipient, changed_self_row)
    self_replacements = {
        source_slot: _with_relation_row(
            shared_case.projection_by_slot[source_slot],
            global_slot=0,
            row=changed_self_row,
            visible=True,
        )
        for source_slot in (1, 2)
    }
    self_accepted = _build(
        shared_case,
        0,
        recipient=matched_self,
        replacements=self_replacements,
    )
    assert self_accepted.scene.agents[0].position == (
        scene.self_actor.position[0] + 0.25,
        scene.self_actor.position[1],
    )

    body_topology = next(
        topology
        for topology in recipient.sensor_source_availability
        if topology.sensor_source_public_agent_id == first_body.public_agent_id
    )
    body_slot = body_topology.sensor_source_global_slot
    original_body_row = shared_case.projection_by_slot[
        body_slot
    ].base_sensor_frame.self_features
    changed_body_row = _changed_feature_row(
        original_body_row,
        AGENT_FEATURE_X_V1,
        original_body_row[AGENT_FEATURE_X_V1] + 0.25,
    )
    matched_body = _with_relation_row(
        recipient,
        global_slot=body_slot,
        row=changed_body_row,
        visible=True,
    )
    body_replacements = {
        source_slot: (
            _with_self_features(
                shared_case.projection_by_slot[source_slot],
                changed_body_row,
            )
            if source_slot == body_slot
            else _with_relation_row(
                shared_case.projection_by_slot[source_slot],
                global_slot=body_slot,
                row=changed_body_row,
                visible=True,
            )
        )
        for source_slot in (1, 2)
    }
    body_accepted = _build(
        shared_case,
        0,
        recipient=matched_body,
        replacements=body_replacements,
    )
    assert next(
        agent
        for agent in body_accepted.scene.agents
        if agent.public_agent_id == first_body.public_agent_id
    ).position == (
        first_body.position[0] + 0.25,
        first_body.position[1],
    )

    lifecycle = recipient.base_sensor_frame.spawn_lifecycle
    pad_positions = [list(team) for team in lifecycle.spawn_pad_positions_by_team]
    original_pad_position = pad_positions[0][0]
    pad_positions[0][0] = (
        original_pad_position[0] + 0.25,
        original_pad_position[1],
    )
    matched_pad = _with_lifecycle(
        recipient,
        spawn_pad_positions_by_team=tuple(tuple(team) for team in pad_positions),
    )
    pad_accepted = _build(shared_case, 0, recipient=matched_pad)
    assert pad_accepted.scene.spawn_pads[0].position == pad_positions[0][0]

    wave_countdowns = list(lifecycle.respawn_wave_countdowns_by_team)
    wave_countdowns[0] -= 1
    matched_wave = _with_lifecycle(
        recipient,
        respawn_wave_countdowns_by_team=tuple(wave_countdowns),
    )
    wave_accepted = _build(shared_case, 0, recipient=matched_wave)
    assert wave_accepted.scene.respawn_waves[0].countdown_steps == wave_countdowns[0]


def test_admitted_contributor_scene_only_map_pad_and_wave_are_inert(
    shared_case: _SharedCase,
) -> None:
    baseline = _build(shared_case, 0)
    contributor = shared_case.projection_by_slot[1]
    scene = contributor.base_sensor_scene
    scene_variants = (
        replace(scene, map=replace(scene.map, width=scene.map.width + 1.0)),
        replace(
            scene,
            spawn_pads=_replace_tuple_item(
                scene.spawn_pads,
                0,
                replace(
                    scene.spawn_pads[0],
                    position=(
                        scene.spawn_pads[0].position[0] + 0.25,
                        scene.spawn_pads[0].position[1],
                    ),
                ),
            ),
        ),
        replace(
            scene,
            respawn_waves=_replace_tuple_item(
                scene.respawn_waves,
                0,
                replace(
                    scene.respawn_waves[0],
                    countdown_steps=scene.respawn_waves[0].countdown_steps - 1,
                ),
            ),
        ),
    )
    for variant in scene_variants:
        forged = copy.copy(contributor)
        object.__setattr__(forged, "base_sensor_scene", variant)
        mutated = _build(shared_case, 0, replacements={1: forged})
        assert _canonical_bytes(mutated) == _canonical_bytes(baseline)


def test_equal_duplicates_merge_with_ordered_provenance_and_input_permutation(
    shared_case: _SharedCase,
) -> None:
    opponent_self = shared_case.projection_by_slot[5].base_sensor_frame.self_features
    source_one = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=opponent_self,
        visible=True,
    )
    source_two = _with_relation_row(
        shared_case.projection_by_slot[2],
        global_slot=5,
        row=opponent_self,
        visible=True,
    )
    sources = _source_tuple(
        shared_case,
        0,
        replacements={1: source_one, 2: source_two},
    )

    first = _build(shared_case, 0, sources=sources)
    permuted = _build(shared_case, 0, sources=tuple(reversed(sources)))

    assert _canonical_bytes(first) == _canonical_bytes(permuted)
    provenance = next(
        row
        for row in first.agent_observation_provenance
        if row.agent_public_agent_id == "agent-slot-5"
    )
    assert tuple(
        source.source_public_agent_id for source in provenance.observation_sources
    ) == ("agent-slot-1", "agent-slot-2")


@pytest.mark.parametrize(
    ("feature_index", "replacement"),
    (
        (AGENT_FEATURE_X_V1, 6.5),
        (AGENT_FEATURE_CURRENT_HEALTH_V1, 90.0),
        (AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1, 1.0),
        (AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1, 1.0),
        (AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1, 1.15),
        (AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1, 999.0),
    ),
)
def test_conflicting_duplicate_fact_families_fail_closed(
    shared_case: _SharedCase,
    feature_index: int,
    replacement: float,
) -> None:
    opponent_self = shared_case.projection_by_slot[5].base_sensor_frame.self_features
    source_one = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=opponent_self,
        visible=True,
    )
    conflicting_row = _changed_feature_row(
        opponent_self,
        feature_index,
        replacement,
    )
    source_two = _with_relation_row(
        shared_case.projection_by_slot[2],
        global_slot=5,
        row=conflicting_row,
        visible=True,
    )

    with pytest.raises(ValueError, match="disagree about one authorized agent"):
        _build(shared_case, 0, replacements={1: source_one, 2: source_two})


def test_unavailable_source_payload_visibility_and_nonidentity_mapping_are_inert(
    shared_case: _SharedCase,
) -> None:
    baseline = _build(shared_case, 0)
    unavailable = shared_case.projection_by_slot[5]
    changed_self = _changed_feature_row(
        unavailable.base_sensor_frame.self_features,
        AGENT_FEATURE_X_V1,
        12.25,
    )
    changed = _with_self_features(unavailable, changed_self)
    changed = _with_relation_row(
        changed,
        global_slot=0,
        row=shared_case.projection_by_slot[0].base_sensor_frame.self_features,
        visible=True,
    )
    axis_payload = changed.axis_mapping.model_dump(mode="python")
    axis_payload["movement_action_name_by_id"] = tuple(
        f"renamed-{index}"
        for index in range(len(changed.axis_mapping.movement_action_name_by_id))
    )
    changed_axis = ActorPovAxisMappingV1.model_validate(axis_payload)
    changed = _with_axis_mapping(changed, changed_axis)

    mutated = _build(shared_case, 0, replacements={5: changed})

    assert _canonical_bytes(mutated) == _canonical_bytes(baseline)


def test_false_availability_ally_payload_is_inert_but_projection_is_required(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    availability = list(recipient.sensor_source_availability)
    availability[2] = replace(availability[2], recorded_available=False)
    recipient = replace(
        recipient,
        sensor_source_availability=tuple(availability),
    )
    baseline = _build(shared_case, 0, recipient=recipient)

    unavailable_ally = shared_case.projection_by_slot[2]
    changed_self = _changed_feature_row(
        unavailable_ally.base_sensor_frame.self_features,
        AGENT_FEATURE_X_V1,
        unavailable_ally.base_sensor_frame.self_features[AGENT_FEATURE_X_V1] + 0.5,
    )
    unavailable_ally = _with_self_features(unavailable_ally, changed_self)
    mutated = _build(
        shared_case,
        0,
        recipient=recipient,
        replacements={2: unavailable_ally},
    )

    assert _canonical_bytes(mutated) == _canonical_bytes(baseline)
    assert "agent-slot-2" not in tuple(
        source.source_public_agent_id for source in baseline.authorized_sensor_sources
    )
    incomplete = tuple(
        source
        for source in _source_tuple(shared_case, 0)
        if source.base_sensor_frame.public_agent_id != "agent-slot-2"
    )
    with pytest.raises(ValueError, match="every and only active nonrecipient"):
        _build(shared_case, 0, recipient=recipient, sources=incomplete)


def test_contributor_public_id_global_slot_remap_rejects_before_filter(
    shared_case: _SharedCase,
) -> None:
    contributor = shared_case.projection_by_slot[5]
    axis_payload = contributor.axis_mapping.model_dump(mode="python")
    ally_ids = list(
        cast(
            tuple[str, ...], axis_payload["ally_observation_row_public_agent_id_by_id"]
        )
    )
    ally_ids[0], ally_ids[1] = ally_ids[1], ally_ids[0]
    axis_payload["ally_observation_row_public_agent_id_by_id"] = tuple(ally_ids)
    enemy_ids = cast(
        tuple[str, ...],
        axis_payload["enemy_observation_row_public_agent_id_by_id"],
    )
    axis_payload["target_action_recipient_public_agent_id_by_id"] = (
        None,
        *ally_ids,
        *enemy_ids,
    )
    remapped_axis = ActorPovAxisMappingV1.model_validate(axis_payload)
    forged = copy.copy(contributor)
    object.__setattr__(forged, "axis_mapping", remapped_axis)

    with pytest.raises(ValueError, match="remaps the canonical actor topology"):
        _build(shared_case, 0, replacements={5: forged})

    forged_self = replace(
        contributor.base_sensor_scene.self_actor,
        team_local_slot=1,
    )
    forged_scene = replace(contributor.base_sensor_scene, self_actor=forged_self)
    forged_header = copy.copy(contributor)
    object.__setattr__(forged_header, "base_sensor_scene", forged_scene)
    with pytest.raises(ValueError, match="self identity does not join topology"):
        _build(shared_case, 0, replacements={5: forged_header})


def test_admitted_contributor_nonunit_branches_are_inert(
    shared_case: _SharedCase,
) -> None:
    baseline = _build(shared_case, 0)
    source = shared_case.projection_by_slot[1]
    variants: list[SharedObsSourceMaterialProjectionV1] = []

    context = list(source.base_sensor_frame.context_features)
    context[CONTEXT_FEATURE_MAP_WIDTH_V1] += 1.0
    variants.append(_with_frame(source, context_features=tuple(context)))

    objectives = list(source.base_sensor_frame.objective_features)
    objective_zero = list(objectives[0])
    objective_zero[0] = 123.0
    objectives[0] = tuple(objective_zero)
    variants.append(_with_frame(source, objective_features=tuple(objectives)))

    variants.append(
        _with_lifecycle(
            source,
            spawn_shield_speed=(
                source.base_sensor_frame.spawn_lifecycle.spawn_shield_speed + 0.5
            ),
        )
    )
    lifecycle = source.base_sensor_frame.spawn_lifecycle
    pad_positions = [list(team) for team in lifecycle.spawn_pad_positions_by_team]
    pad_positions[1][0] = (11.0, 3.0)
    alive = [list(team) for team in lifecycle.alive_mask_by_team]
    alive[1][0] = False
    shields = [list(team) for team in lifecycle.spawn_shield_actual_durations_by_team]
    shields[0][1] = 2
    shields[1][0] = 0
    variants.append(
        _with_lifecycle(
            source,
            spawn_pad_positions_by_team=tuple(tuple(team) for team in pad_positions),
            alive_mask_by_team=tuple(tuple(team) for team in alive),
            spawn_shield_actual_durations_by_team=tuple(
                tuple(team) for team in shields
            ),
        )
    )
    variants.append(
        _with_frame(
            source,
            action_mask=shared_case.projection_by_slot[2].base_sensor_frame.action_mask,
        )
    )
    variants.append(
        _with_frame(
            source,
            previous_timestep_actions=shared_case.projection_by_slot[
                2
            ].base_sensor_frame.previous_timestep_actions,
        )
    )
    availability = list(source.sensor_source_availability)
    ally_index = next(
        index
        for index, row in enumerate(availability)
        if row.relation_to_recipient == "ally" and row.recorded_available
    )
    availability[ally_index] = replace(
        availability[ally_index],
        recorded_available=False,
    )
    variants.append(replace(source, sensor_source_availability=tuple(availability)))

    for variant in variants:
        mutated = _build(shared_case, 0, replacements={1: variant})
        assert _canonical_bytes(mutated) == _canonical_bytes(baseline)


def test_authorized_visible_fact_and_visibility_entry_change_scene_bytes(
    shared_case: _SharedCase,
) -> None:
    baseline = _build(shared_case, 0)
    opponent_self = shared_case.projection_by_slot[5].base_sensor_frame.self_features
    visible = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=opponent_self,
        visible=True,
    )
    entered = _build(shared_case, 0, replacements={1: visible})
    changed_row = _changed_feature_row(
        opponent_self,
        AGENT_FEATURE_X_V1,
        opponent_self[AGENT_FEATURE_X_V1] - 0.25,
    )
    changed_visible = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=changed_row,
        visible=True,
    )
    changed = _build(shared_case, 0, replacements={1: changed_visible})

    assert _canonical_bytes(entered) != _canonical_bytes(baseline)
    assert _canonical_bytes(changed) != _canonical_bytes(entered)


def test_recipient_owns_map_lifecycle_mask_while_history_is_inert(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    baseline = _build(shared_case, 0)

    context = list(recipient.base_sensor_frame.context_features)
    context[CONTEXT_FEATURE_MAP_WIDTH_V1] += 1.0
    changed_map = _build(
        shared_case,
        0,
        recipient=_with_frame(recipient, context_features=tuple(context)),
    )
    changed_lifecycle = _build(
        shared_case,
        0,
        recipient=_with_lifecycle(
            recipient,
            spawn_shield_speed=(
                recipient.base_sensor_frame.spawn_lifecycle.spawn_shield_speed + 0.5
            ),
        ),
    )
    changed_mask = _build(
        shared_case,
        0,
        recipient=_with_frame(
            recipient,
            action_mask=shared_case.projection_by_slot[2].base_sensor_frame.action_mask,
        ),
    )
    changed_history = _build(
        shared_case,
        0,
        recipient=_with_frame(
            recipient,
            previous_timestep_actions=shared_case.projection_by_slot[
                2
            ].base_sensor_frame.previous_timestep_actions,
        ),
    )
    objectives = list(recipient.base_sensor_frame.objective_features)
    first_objective = list(objectives[0])
    first_objective[0] = 321.0
    objectives[0] = tuple(first_objective)
    changed_objectives = _build(
        shared_case,
        0,
        recipient=_with_frame(
            recipient,
            objective_features=tuple(objectives),
        ),
    )

    assert changed_map.scene.map.width == baseline.scene.map.width + 1.0
    assert _canonical_bytes(changed_lifecycle) != _canonical_bytes(baseline)
    assert changed_mask.next_decision_action_mask != (
        baseline.next_decision_action_mask
    )
    assert _canonical_bytes(changed_history) == _canonical_bytes(baseline)
    assert _canonical_bytes(changed_objectives) == _canonical_bytes(baseline)


def test_team_b_recipient_lifecycle_remaps_payloads_to_absolute_teams(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[5]
    lifecycle = recipient.base_sensor_frame.spawn_lifecycle
    positions = [list(team) for team in lifecycle.spawn_pad_positions_by_team]
    positions[0][0] = (17.0, 9.0)
    positions[1][0] = (3.0, 2.0)
    shields = [list(team) for team in lifecycle.spawn_shield_actual_durations_by_team]
    shields[0][0] = 2
    shields[1][0] = 1
    changed = _with_lifecycle(
        recipient,
        spawn_pad_positions_by_team=tuple(tuple(team) for team in positions),
        spawn_shield_actual_durations_by_team=tuple(tuple(team) for team in shields),
        respawn_wave_period_step_count_by_team=(17, 13),
        respawn_wave_countdowns_by_team=(7, 3),
    )

    parts = _build(shared_case, 5, recipient=changed)

    team_a_pad = parts.scene.spawn_pads[0]
    team_b_pad = parts.scene.spawn_pads[5]
    assert (team_a_pad.position, team_a_pad.spawn_shield_remaining) == (
        (3.0, 2.0),
        1,
    )
    assert (team_b_pad.position, team_b_pad.spawn_shield_remaining) == (
        (17.0, 9.0),
        2,
    )
    assert tuple(
        (wave.team_id, wave.period_steps, wave.countdown_steps)
        for wave in parts.scene.respawn_waves
    ) == ((1, 13, 3), (2, 17, 7))


def test_recipient_scoped_opaque_keys_are_stable_and_oracle_distinct(
    shared_case: _SharedCase,
) -> None:
    first = _build(shared_case, 0, authority_session_id="stable-authority")
    next_projections = {
        slot: build_shared_obs_source_material_projection_v1(
            shared_case.trajectory.context,
            shared_case.trajectory.frames[1],
            selected_global_slot=slot,
            transition_view=adapter.EvaluationTransitionViewV1(
                context=shared_case.trajectory.context,
                start_frame=shared_case.trajectory.frames[0],
                transition=shared_case.trajectory.transitions[0],
                successor_frame=shared_case.trajectory.frames[1],
            ),
        )
        for slot in shared_case.active_slots
    }
    second = build_shared_obs_authorized_scene_v1(
        next_projections[0],
        all_active_nonrecipient_source_material=tuple(
            next_projections[slot] for slot in shared_case.active_slots if slot != 0
        ),
        public_catalog=shared_case.trajectory.context.static_mechanics_catalog,
        authority_session_id="stable-authority",
    )
    other_recipient = _build(
        shared_case,
        5,
        authority_session_id="stable-authority",
    )

    first_keys = {
        agent.public_agent_id: agent.presentation_key for agent in first.scene.agents
    }
    second_keys = {
        agent.public_agent_id: agent.presentation_key for agent in second.scene.agents
    }
    assert first_keys == second_keys
    assert (
        first.recipient_presentation_key != other_recipient.recipient_presentation_key
    )
    for public_id, key in first_keys.items():
        assert key.startswith("pov_") and len(key) == 68
        assert public_id not in key
        assert not key.startswith("oracle_")
    assert (
        first.authorized_sensor_sources[1].source_presentation_key
        == first_keys[first.authorized_sensor_sources[1].source_public_agent_id]
    )


def test_cross_class_status_uses_catalog_and_unrepresented_source_semantics(
    shared_case: _SharedCase,
) -> None:
    opponent = shared_case.projection_by_slot[5].base_sensor_frame.self_features
    status_row = _changed_feature_row(opponent, 24, 1.0)
    status_row = _changed_feature_row(
        status_row,
        25,
        shared_case.trajectory.context.static_mechanics_catalog.status_channels[
            6
        ].magnitude
        or 0.0,
    )
    source = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=status_row,
        visible=True,
    )

    parts = _build(shared_case, 0, replacements={1: source})
    target = next(
        agent for agent in parts.scene.agents if agent.public_agent_id == "agent-slot-5"
    )
    status = next(row for row in target.statuses if row.status_channel == 6)

    assert status.source_class_name == "Rogue"
    assert status.direct_sources == ()
    assert 4 not in tuple(row.class_id for row in parts.scene.class_mechanics)
    assert status.configured_duration_steps == (
        shared_case.trajectory.context.static_mechanics_catalog.status_channels[
            6
        ].duration_steps
    )


def test_authorized_aura_fields_hidden_emitter_noncoupling_and_neutral_omission(
    shared_case: _SharedCase,
) -> None:
    team_a = _build(shared_case, 0)
    assert tuple(
        (field.source_public_agent_id, field.aura_id)
        for field in team_a.scene.aura_fields
    ) == (
        ("agent-slot-0", "mage_damage_amplification"),
        ("agent-slot-1", "warrior_damage_mitigation"),
    )
    assert all(
        modifier.multiplier != 1.0
        for agent in team_a.scene.agents
        for modifier in agent.aura_modifiers
    )

    team_b_recipient = shared_case.projection_by_slot[5]
    nonneutral = _changed_feature_row(
        team_b_recipient.base_sensor_frame.self_features,
        AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
        1.15,
    )
    team_b_recipient = _with_self_features(team_b_recipient, nonneutral)
    source_six = _with_relation_row(
        shared_case.projection_by_slot[6],
        global_slot=5,
        row=nonneutral,
        visible=True,
    )
    team_b = _build(
        shared_case,
        5,
        recipient=team_b_recipient,
        replacements={6: source_six},
    )
    owner = next(agent for agent in team_b.scene.agents if agent.relation == "self")

    assert tuple(
        (modifier.aura_id, modifier.multiplier) for modifier in owner.aura_modifiers
    ) == (("mage_damage_amplification", 1.15),)
    assert all(
        field.aura_id != "mage_damage_amplification"
        for field in team_b.scene.aura_fields
    )


def test_catalog_unrepresented_revokes_profile_and_represented_clamp_sensitive(
    shared_case: _SharedCase,
) -> None:
    catalog = shared_case.trajectory.context.static_mechanics_catalog
    baseline = _build(shared_case, 0)

    unrepresented_payload = catalog.model_dump(mode="python")
    class_rows = list(
        cast(tuple[dict[str, object], ...], unrepresented_payload["class_mechanics"])
    )
    hunter = dict(class_rows[3])
    hunter["basic_raw_damage"] = cast(float, hunter["basic_raw_damage"]) + 1.0
    class_rows[3] = hunter
    unrepresented_payload["class_mechanics"] = tuple(class_rows)
    unrepresented = _catalog_from_payload(unrepresented_payload)
    profile_revoked = _build(shared_case, 0, catalog=unrepresented)

    assert 3 not in {row.class_id for row in baseline.scene.class_mechanics}
    baseline_mechanics = tuple(
        cast(AuthorizedClassMechanicsV2, row) for row in baseline.scene.class_mechanics
    )
    revoked_mechanics = tuple(
        cast(AuthorizedClassMechanicsV2, row)
        for row in profile_revoked.scene.class_mechanics
    )
    assert all(
        type(row.documentation_profile)
        is AuthorizedClassDocumentationProfileAvailableV1
        for row in baseline_mechanics
    )
    assert all(
        type(row.documentation_profile)
        is AuthorizedClassDocumentationProfileUnavailableV1
        for row in revoked_mechanics
    )
    baseline_profile_by_class = {
        row.class_id: row.documentation_profile for row in baseline_mechanics
    }
    restored_profile = replace(
        profile_revoked,
        scene=replace(
            profile_revoked.scene,
            class_mechanics=tuple(
                replace(
                    row,
                    documentation_profile=baseline_profile_by_class[row.class_id],
                )
                for row in revoked_mechanics
            ),
        ),
    )
    assert _canonical_bytes(profile_revoked) != _canonical_bytes(baseline)
    assert _canonical_bytes(restored_profile) == _canonical_bytes(baseline)

    represented_payload = catalog.model_dump(mode="python")
    aura_rows = list(
        cast(tuple[dict[str, object], ...], represented_payload["aura_mechanics"])
    )
    mage_aura = dict(aura_rows[0])
    mage_aura["clamp_value"] = 1.4
    aura_rows[0] = mage_aura
    represented_payload["aura_mechanics"] = tuple(aura_rows)
    represented = _catalog_from_payload(represented_payload)
    changed = _build(shared_case, 0, catalog=represented)

    assert _canonical_bytes(changed) != _canonical_bytes(baseline)
    assert changed.scene.aura_fields[0].clamp_value == 1.4

    forged = catalog.model_copy(update={"canonical_digest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="canonical digest mismatch"):
        _build(shared_case, 0, catalog=forged)


def test_admitted_dead_source_self_allowed_but_dead_relation_row_rejected(
    shared_case: _SharedCase,
) -> None:
    recipient = shared_case.projection_by_slot[0]
    lifecycle = recipient.base_sensor_frame.spawn_lifecycle
    alive = [list(team) for team in lifecycle.alive_mask_by_team]
    alive[0][1] = False
    recipient = _with_lifecycle(
        recipient,
        alive_mask_by_team=tuple(tuple(team) for team in alive),
    )
    recipient = _with_relation_row(
        recipient,
        global_slot=1,
        row=recipient.base_sensor_frame.ally_unit_features[1],
        visible=False,
    )
    source_one = shared_case.projection_by_slot[1]
    dead_self = _changed_feature_row(
        source_one.base_sensor_frame.self_features,
        AGENT_FEATURE_ALIVE_V1,
        0.0,
    )
    dead_self = _changed_feature_row(
        dead_self,
        AGENT_FEATURE_CURRENT_HEALTH_V1,
        0.0,
    )
    source_one = _with_self_features(source_one, dead_self)
    source_two = _with_relation_row(
        shared_case.projection_by_slot[2],
        global_slot=1,
        row=shared_case.projection_by_slot[1].base_sensor_frame.self_features,
        visible=False,
    )

    accepted = _build(
        shared_case,
        0,
        recipient=recipient,
        replacements={1: source_one, 2: source_two},
    )
    dead_source = next(
        agent
        for agent in accepted.scene.agents
        if agent.public_agent_id == "agent-slot-1"
    )
    assert dead_source.life_state == "corpse"

    opponent_dead = _changed_feature_row(
        shared_case.projection_by_slot[5].base_sensor_frame.self_features,
        AGENT_FEATURE_ALIVE_V1,
        0.0,
    )
    opponent_dead = _changed_feature_row(
        opponent_dead,
        AGENT_FEATURE_CURRENT_HEALTH_V1,
        0.0,
    )
    recipient_lifecycle = shared_case.projection_by_slot[
        0
    ].base_sensor_frame.spawn_lifecycle
    opponent_alive = [list(team) for team in recipient_lifecycle.alive_mask_by_team]
    opponent_alive[1][0] = False
    recipient_with_dead_opponent = _with_lifecycle(
        shared_case.projection_by_slot[0],
        alive_mask_by_team=tuple(tuple(team) for team in opponent_alive),
    )
    dead_relation = _with_relation_row(
        shared_case.projection_by_slot[1],
        global_slot=5,
        row=opponent_dead,
        visible=True,
    )
    with pytest.raises(ValueError, match="visible relation rows must be alive"):
        _build(
            shared_case,
            0,
            recipient=recipient_with_dead_opponent,
            replacements={1: dead_relation},
        )


def _json_tree_contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return any(
            key in forbidden or _json_tree_contains_key(child, forbidden)
            for key, child in mapping.items()
        )
    if isinstance(value, list):
        items = cast(list[object], value)
        return any(_json_tree_contains_key(child, forbidden) for child in items)
    return False


def test_shared_and_no_shared_parts_are_strict_json_endpoint_roots(
    shared_case: _SharedCase,
) -> None:
    shared = _build(shared_case, 0)
    no_shared_trajectory = captured_evaluation_trajectory(transition_count=0)
    current_slice = build_actor_pov_current_slice_v1(
        no_shared_trajectory.context,
        no_shared_trajectory.frames[0],
        global_slot=0,
    )
    no_shared = build_no_shared_obs_authorized_scene_v1(
        current_slice,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="strict-no-shared",
    )

    for root_type, parts in (
        (SharedObsAuthorizedScenePartsV1, shared),
        (NoSharedObsAuthorizedScenePartsV1, no_shared),
    ):
        root_adapter = TypeAdapter(root_type)
        encoded = root_adapter.dump_json(parts)
        assert root_adapter.validate_json(encoded) == parts
        payload = json.loads(encoded)

        coerced = dict(payload)
        coerced["source_frame_index"] = str(payload["source_frame_index"])
        with pytest.raises(ValidationError):
            root_adapter.validate_json(json.dumps(coerced))

        missing = dict(payload)
        missing.pop("source_simulator_step_count")
        with pytest.raises(ValidationError):
            root_adapter.validate_json(json.dumps(missing))

        nested_extra = dict(payload)
        nested_extra["scene"] = dict(payload["scene"])
        nested_extra["scene"]["agents"] = list(payload["scene"]["agents"])
        nested_extra["scene"]["agents"][0] = dict(nested_extra["scene"]["agents"][0])
        nested_extra["scene"]["agents"][0]["global_slot"] = 0
        with pytest.raises(ValidationError):
            root_adapter.validate_json(json.dumps(nested_extra))

    assert type(shared.authorized_sensor_sources[0]) is (
        SharedObsAuthorizedSensorSourceV1
    )
    assert type(shared.agent_observation_provenance[0]) is (
        SharedObsAgentObservationProvenanceV1
    )
    payload = json.loads(TypeAdapter(SharedObsAuthorizedScenePartsV1).dump_json(shared))
    forbidden_keys = {
        "global_slot",
        "source_frame_id",
        "source_material_frame_id",
        "incoming_transition_id",
        "incoming_event_ids",
        "previous_timestep_actions",
        "reward",
        "metric",
    }
    assert not _json_tree_contains_key(payload, forbidden_keys)
    serialized = json.dumps(payload, sort_keys=True)
    for privileged_value in (
        "PRIVILEGED RESEARCHER",
        "oracle_",
        "canonical-task-reward",
        "policy-0",
    ):
        assert privileged_value not in serialized


@pytest.mark.parametrize("poison_relation", ("oracle", "self", "opponent"))
def test_shared_root_rejects_relation_or_key_rewrites(
    shared_case: _SharedCase,
    poison_relation: str,
) -> None:
    parts = _build(shared_case, 0)
    root_adapter = TypeAdapter(SharedObsAuthorizedScenePartsV1)
    payload = json.loads(root_adapter.dump_json(parts))
    payload["scene"]["agents"][1]["relation"] = poison_relation
    with pytest.raises((ValidationError, ValueError)):
        root_adapter.validate_json(json.dumps(payload))

    key_payload = json.loads(root_adapter.dump_json(parts))
    key_payload["recipient_presentation_key"] = "oracle_" + "0" * 64
    with pytest.raises((ValidationError, ValueError)):
        root_adapter.validate_json(json.dumps(key_payload))


def test_shared_builder_never_calls_oracle_or_no_shared_scene_helpers(
    shared_case: _SharedCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.rendering.authorized_pov_scene as module

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Shared union must not call the NoShared/Oracle helper")

    monkeypatch.setattr(module, "build_actor_pov_analyzer_projection_v1", forbidden)
    parts = _build(shared_case, 0)
    assert parts.recipient_public_agent_id == "agent-slot-0"

    module_path = (
        _REPOSITORY_ROOT / "src/marl_battlegrounds/rendering/authorized_pov_scene.py"
    )
    module_source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    shared_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_shared_obs_authorized_scene_v1"
    )
    parameter_names = {
        argument.arg
        for argument in (
            *shared_function.args.posonlyargs,
            *shared_function.args.args,
            *shared_function.args.kwonlyargs,
        )
    }
    assert parameter_names == {
        "recipient_source_material",
        "all_active_nonrecipient_source_material",
        "public_catalog",
        "authority_session_id",
    }
    call_names = {
        node.func.id
        for node in ast.walk(shared_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {name for name in call_names if "oracle" in name.lower()}
    assert "build_actor_pov_analyzer_projection_v1" not in call_names

    assert "incoming_transition_id" not in module_source
    assert "_base_sensor_map_scene" not in module_source
    post_init_receivers = {
        ast.unparse(node.func.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__post_init__"
    }
    assert post_init_receivers == {"row"}


def test_shared_builder_imports_without_jax_numpy_or_core() -> None:
    code = """
import sys
import marl_battlegrounds.rendering.authorized_pov_scene
forbidden = ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core')
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
print('forbidden', loaded)
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cuda"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "forbidden []"


def test_shared_public_exports_are_exact() -> None:
    import marl_battlegrounds.rendering as rendering

    assert rendering.SharedObsAuthorizedScenePartsV1 is (
        SharedObsAuthorizedScenePartsV1
    )
    assert rendering.SharedObsAuthorizedSensorSourceV1 is (
        SharedObsAuthorizedSensorSourceV1
    )
    assert rendering.SharedObsAgentObservationProvenanceV1 is (
        SharedObsAgentObservationProvenanceV1
    )
    assert rendering.build_shared_obs_authorized_scene_v1 is (
        build_shared_obs_authorized_scene_v1
    )
