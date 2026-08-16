"""Canonical synthetic fixtures for deterministic renderer/browser tests.

These records are deliberately not simulator scenarios. They never construct
or submit actions, and they must never be presented as scientific history.
Researcher fixtures use the exact Scene/Event V2 roots consumed in production;
the POV fixture is independently recipient-sliced and never filters a
researcher scene in the browser.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from pydantic import BaseModel

from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovEpisodeEndedCueV1,
    ActorPovOwnActionOutcomeCueV1,
    ActorPovOwnCooldownChangedCueV1,
    ActorPovOwnHealthChangedCueV1,
    ActorPovOwnLifecycleChangedCueV1,
    ActorPovOwnPositionChangedCueV1,
    ActorPovOwnStatusChangedCueV1,
    ActorPovVisibleBodyObservationChangedCueV1,
)
from marl_battlegrounds.evaluation.replay import ReplayArtifactReferenceV1
from marl_battlegrounds.rendering.pov_scene import (
    ACTOR_POV_SCENE_SCHEMA_VERSION,
    ActorPovAnalyzerProjectionV1,
    ActorPovBattlefieldSceneV1,
    ActorPovRespawnWaveSceneV1,
    ActorPovSelfSceneV1,
    ActorPovSpawnPadSceneV1,
    ActorPovVisibleBodySceneV1,
)
from marl_battlegrounds.rendering.scene import (
    EVENT_V2_SCHEMA_VERSION,
    SCENE_V2_SCHEMA_VERSION,
    AbilityActivatedEventV2,
    ActionRejectedEventV2,
    AgentDiedEventV2,
    AgentRespawnedEventV2,
    AgentSceneV2,
    AuraFieldSceneV2,
    AuraRecipientModifierSceneV2,
    BasicTargetModeV2,
    BattlefieldSceneV2,
    ChargePhaseDisplacementEventV2,
    ClassAuraMechanicSceneV2,
    ClassMechanicsSceneV2,
    ClassStatusMechanicSceneV2,
    CombatCountdownResetEventV2,
    CooldownReadyEventV2,
    CooldownStartedEventV2,
    HealthRegeneratedEventV2,
    LethalDamageContributionEventV2,
    MapSceneV1,
    ObserverVisibilitySceneV1,
    ObstacleSceneV1,
    OrdinaryMovementPhaseDisplacementEventV2,
    RangeSceneV1,
    RecipientHealthResolutionEventV2,
    ResearcherAnalyzerProjectionV2,
    RespawnWaveOccurredEventV2,
    RespawnWaveSceneV2,
    SelectedLegalitySceneV1,
    SelectionSceneV1,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    SpawnPadSceneV2,
    SpawnShieldExpiredEventV2,
    StatusAgedToZeroEventV2,
    StatusAppliedEventV2,
    StatusBrokenByDamageEventV2,
    StatusClearedByNewDeathEventV2,
    StatusFamilyV2,
    StatusMagnitudeKindV2,
    StatusRefreshedOrExtendedEventV2,
    StatusSceneV2,
    StatusSourceChannelEvidenceV2,
    StatusSourceEvidenceStateV2,
    TeamDeathmatchCompletedEventV2,
    TeamDeathmatchScoreChangedEventV2,
    UltimateTargetModeV2,
    VisualAgentAnchorV2,
    VisualAgentPhaseTrajectoryV2,
    VisualAnchorPhaseV2,
    VisualEventBatchV2,
    VisualEventV2,
    VisualTeamAnchorV2,
)
from marl_battlegrounds.rendering.vocabulary import (
    CANONICAL_STATUS_ORDER,
    CATALOG_STATUS_ID_BY_CHANNEL,
    StatusTokenId,
    lookup_status_token,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovCandidateLegalityCardV1,
    ActorPovHudFrameV1,
    ActorPovLiveDebuggerFrameV2,
    ActorPovPendingActionCardV1,
    ActorPovTargetReferenceV1,
    MovementLegalityCardV1,
    PendingActionCardV1,
    ResearcherHudFrameV2,
    ResearcherLiveDebuggerFrameV2,
    ScenarioMetadataV1,
    ScenarioOptionV1,
    TargetReferenceV1,
    TerminalStateV2,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
    ActorPovProcessingDisclosureV1,
    ActorPovReplayCompletionBadgeV1,
    ActorPovReplayTimelineRowV1,
    ActorPovReplayTimelineV1,
    ActorPovReplayViewerFrameV1,
    ReplayArtifactSummaryV1,
    ReplayCompletionBadgeV1,
    ReplayCursorV1,
    ReplayProcessingBadgeV1,
    ResearcherReplayTimelineRowV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
)

type RendererFixtureName = Literal[
    "visual_vocabulary",
    "durable_controls",
    "crowded_teamfight",
    "required_dock_fallback",
    "route_collision",
    "mixed_net_zero",
    "viewport_matrix",
    "canonical_event_vocabulary",
    "pov_redaction",
]
type ViewportLabel = Literal["desktop", "compact", "minimum", "stacked"]
type ViewportLayout = Literal["split", "stacked"]
type AuraIdV2 = Literal[
    "mage_damage_amplification",
    "warrior_damage_mitigation",
]
type CatalogStatusId = Literal[
    "warrior_charge_slow",
    "hunter_basic_slow",
    "rogue_poison_slow",
    "warrior_charge_stun",
    "hunter_trap_stun",
    "rogue_poison_stun",
    "rogue_poison_anti_heal",
    "mage_burst_damage_amplification",
    "priest_blessing_of_freedom_movement_floor",
]
type RendererScene = BattlefieldSceneV2 | ActorPovBattlefieldSceneV1
type RendererLiveFrame = ResearcherLiveDebuggerFrameV2 | ActorPovLiveDebuggerFrameV2
type RendererReplayFrame = ResearcherReplayViewerFrameV1 | ActorPovReplayViewerFrameV1
type RendererReplayTimeline = ResearcherReplayTimelineV1 | ActorPovReplayTimelineV1


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewportCaseV1:
    """One deterministic browser viewport and expected responsive layout."""

    label: ViewportLabel
    width: int
    height: int
    expected_layout: ViewportLayout

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("viewport label must be a non-empty Python string.")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("viewport width must be a positive Python int.")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("viewport height must be a positive Python int.")
        if self.expected_layout not in ("split", "stacked"):
            raise ValueError("expected_layout must be 'split' or 'stacked'.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticFixturePresentationPair:
    """Fixture-only live/replay envelopes; this is not a replay wire schema."""

    audience: Literal["researcher", "agent_pov"]
    live_frame: RendererLiveFrame
    replay_frame: RendererReplayFrame
    replay_timeline: RendererReplayTimeline

    def __post_init__(self) -> None:
        if self.audience == "researcher":
            if (
                type(self.live_frame) is not ResearcherLiveDebuggerFrameV2
                or type(self.replay_frame) is not ResearcherReplayViewerFrameV1
                or type(self.replay_timeline) is not ResearcherReplayTimelineV1
            ):
                raise ValueError(
                    "researcher presentation pairs require exact researcher roots."
                )
        elif self.audience == "agent_pov":
            if (
                type(self.live_frame) is not ActorPovLiveDebuggerFrameV2
                or type(self.replay_frame) is not ActorPovReplayViewerFrameV1
                or type(self.replay_timeline) is not ActorPovReplayTimelineV1
            ):
                raise ValueError("POV presentation pairs require exact POV roots.")
        else:
            raise ValueError("unknown presentation-pair audience.")

        if self.live_frame.projection != self.replay_frame.projection:
            raise ValueError(
                "live and replay envelopes must share one exact projection root."
            )
        if (
            self.replay_timeline.artifact_summary != self.replay_frame.artifact_summary
            or self.replay_timeline.timeline_id != self.replay_frame.timeline_id
            or self.replay_timeline.final_frame_index
            != self.replay_frame.cursor.final_frame_index
        ):
            raise ValueError("synthetic replay frame and timeline must join exactly.")
        selected_row = self.replay_timeline.rows[self.replay_frame.cursor.frame_index]
        if selected_row.simulator_step_count != self.replay_frame.simulator_step_count:
            raise ValueError("synthetic replay row must join the selected frame epoch.")
        if self.audience == "researcher":
            researcher_live = cast(ResearcherLiveDebuggerFrameV2, self.live_frame)
            researcher_replay = cast(
                ResearcherReplayViewerFrameV1,
                self.replay_frame,
            )
            researcher_row = cast(ResearcherReplayTimelineRowV1, selected_row)
            incoming_events = researcher_live.projection.incoming_events
            if (
                incoming_events is None
                or researcher_row.frame_id != researcher_replay.frame_id
                or researcher_row.incoming_transition_id
                != researcher_replay.incoming_transition_id
                or researcher_row.incoming_event_count != len(incoming_events.events)
            ):
                raise ValueError(
                    "synthetic researcher timeline rows must join frame and event "
                    "counts exactly."
                )
        else:
            pov_live = cast(ActorPovLiveDebuggerFrameV2, self.live_frame)
            pov_replay = cast(ActorPovReplayViewerFrameV1, self.replay_frame)
            pov_row = cast(ActorPovReplayTimelineRowV1, selected_row)
            if (
                pov_row.pov_frame_id != pov_replay.pov_frame_id
                or pov_row.incoming_pov_transition_id
                != pov_replay.incoming_pov_transition_id
                or pov_row.incoming_cue_count != len(pov_live.projection.incoming_cues)
            ):
                raise ValueError(
                    "synthetic POV timeline rows must join frame and cue counts "
                    "exactly."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class RendererFixtureV2:
    """One explicitly synthetic, audience-specific presentation payload."""

    name: RendererFixtureName
    description: str
    audience: Literal["researcher", "agent_pov"]
    scene: RendererScene
    live_frame: RendererLiveFrame
    event_batch: VisualEventBatchV2 | None = None
    pov_projection: ActorPovAnalyzerProjectionV1 | None = None
    pov_target_public_agent_ids: tuple[str | None, ...] | None = None
    viewports: tuple[ViewportCaseV1, ...] = ()
    exercise_reduced_motion: bool = False
    privileged_source_scene: BattlefieldSceneV2 | None = None
    privileged_source_event_batch: VisualEventBatchV2 | None = None
    synthetic_presentation_pair: SyntheticFixturePresentationPair | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("fixture name must be a non-empty Python string.")
        if type(self.description) is not str or not self.description.startswith(
            "SYNTHETIC:"
        ):
            raise ValueError("fixture description must begin with 'SYNTHETIC:'.")
        if type(self.viewports) is not tuple or any(
            type(viewport) is not ViewportCaseV1 for viewport in self.viewports
        ):
            raise ValueError("viewports must contain exact ViewportCaseV1 rows.")
        labels = tuple(viewport.label for viewport in self.viewports)
        if len(labels) != len(set(labels)):
            raise ValueError("viewport labels must be unique.")
        if type(self.exercise_reduced_motion) is not bool:
            raise ValueError("exercise_reduced_motion must be a Python bool.")

        if self.audience == "researcher":
            if type(self.scene) is not BattlefieldSceneV2:
                raise ValueError("researcher fixtures require BattlefieldSceneV2.")
            if self.pov_projection is not None:
                raise ValueError("researcher fixtures cannot carry a POV projection.")
            if self.pov_target_public_agent_ids is not None:
                raise ValueError("researcher fixtures cannot carry a POV target axis.")
            if (
                type(self.live_frame) is not ResearcherLiveDebuggerFrameV2
                or self.live_frame.projection.scene != self.scene
                or self.live_frame.projection.incoming_events != self.event_batch
            ):
                raise ValueError(
                    "researcher fixtures require their exact validated V2 frame."
                )
        elif self.audience == "agent_pov":
            if type(self.scene) is not ActorPovBattlefieldSceneV1:
                raise ValueError("POV fixtures require the exact recipient scene.")
            if self.event_batch is not None:
                raise ValueError("POV fixtures cannot carry researcher event batches.")
            if (
                type(self.pov_projection) is not ActorPovAnalyzerProjectionV1
                or self.pov_projection.scene != self.scene
            ):
                raise ValueError("POV fixtures require their exact local projection.")
            target_axis = self.pov_target_public_agent_ids
            if (
                type(target_axis) is not tuple
                or len(target_axis) != 11
                or target_axis[0] is not None
                or any(type(value) is not str or not value for value in target_axis[1:])
                or len(set(target_axis[1:])) != 10
            ):
                raise ValueError(
                    "POV fixtures require one exact recipient-local target axis."
                )
            if (
                type(self.live_frame) is not ActorPovLiveDebuggerFrameV2
                or self.live_frame.projection != self.pov_projection
            ):
                raise ValueError(
                    "POV fixtures require their exact validated recipient frame."
                )
        else:
            raise ValueError("unknown fixture audience.")

        if (self.privileged_source_scene is None) != (
            self.privileged_source_event_batch is None
        ):
            raise ValueError(
                "privileged source scene and event batch must be supplied together."
            )
        if self.privileged_source_scene is not None:
            if self.audience != "agent_pov":
                raise ValueError("only POV fixtures may carry a comparison source.")
            if type(self.privileged_source_scene) is not BattlefieldSceneV2:
                raise ValueError("privileged comparison must use BattlefieldSceneV2.")
            if type(self.privileged_source_event_batch) is not VisualEventBatchV2:
                raise ValueError("privileged comparison must use VisualEventBatchV2.")

        pair = self.synthetic_presentation_pair
        if pair is not None:
            if type(pair) is not SyntheticFixturePresentationPair:
                raise ValueError(
                    "synthetic_presentation_pair must be an exact "
                    "SyntheticFixturePresentationPair."
                )
            if pair.audience != self.audience:
                raise ValueError("presentation pair audience must match its fixture.")
            if pair.live_frame.projection.scene != self.scene:
                raise ValueError(
                    "presentation pair must reuse the fixture's exact scene root."
                )
            if self.audience == "researcher":
                researcher_pair = cast(
                    ResearcherLiveDebuggerFrameV2,
                    pair.live_frame,
                )
                if researcher_pair.projection.incoming_events != self.event_batch:
                    raise ValueError(
                        "researcher presentation pair must reuse the exact event batch."
                    )


_CLASS_IDS = (
    MAGE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
    MAGE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
)
_TEAM_IDS = (TEAM_A_ID,) * 5 + (TEAM_B_ID,) * 5
_CLASS_NAMES = {
    MAGE_CLASS_ID: "Mage",
    WARRIOR_CLASS_ID: "Warrior",
    HUNTER_CLASS_ID: "Hunter",
    ROGUE_CLASS_ID: "Rogue",
    PRIEST_CLASS_ID: "Priest",
}
_STATUS_DETAILS: Mapping[
    StatusTokenId,
    tuple[
        StatusFamilyV2,
        Literal["basic", "ultimate"],
        StatusMagnitudeKindV2,
        float | None,
        bool,
    ],
] = MappingProxyType(
    {
        "stun_warrior_charge": ("stun", "ultimate", "none", None, False),
        "stun_hunter_trap": ("stun", "ultimate", "none", None, True),
        "stun_rogue_poison": ("stun", "ultimate", "none", None, False),
        "slow_warrior_charge": (
            "slow",
            "ultimate",
            "movement_multiplier",
            0.5,
            False,
        ),
        "slow_hunter_basic": (
            "slow",
            "basic",
            "movement_multiplier",
            0.8,
            False,
        ),
        "slow_rogue_poison": (
            "slow",
            "ultimate",
            "movement_multiplier",
            0.6,
            False,
        ),
        "anti_heal_rogue_poison": (
            "anti_heal",
            "ultimate",
            "healing_multiplier",
            0.0,
            False,
        ),
        "priest_freedom": (
            "movement_floor",
            "basic",
            "movement_floor",
            1.0,
            False,
        ),
        "mage_burst": (
            "damage_amplification",
            "ultimate",
            "damage_multiplier",
            1.5,
            False,
        ),
    }
)
_CATALOG_STATUS_ID_BY_TOKEN_ID: Mapping[StatusTokenId, CatalogStatusId] = (
    MappingProxyType(
        {
            "slow_warrior_charge": "warrior_charge_slow",
            "slow_hunter_basic": "hunter_basic_slow",
            "slow_rogue_poison": "rogue_poison_slow",
            "stun_warrior_charge": "warrior_charge_stun",
            "stun_hunter_trap": "hunter_trap_stun",
            "stun_rogue_poison": "rogue_poison_stun",
            "anti_heal_rogue_poison": "rogue_poison_anti_heal",
            "mage_burst": "mage_burst_damage_amplification",
            "priest_freedom": "priest_blessing_of_freedom_movement_floor",
        }
    )
)
CATALOG_STATUS_ORDER: tuple[CatalogStatusId, ...] = tuple(
    _CATALOG_STATUS_ID_BY_TOKEN_ID[token_id] for token_id in CANONICAL_STATUS_ORDER
)
_STATUS_DURATION_BY_TOKEN_ID: Mapping[StatusTokenId, int] = MappingProxyType(
    dict(zip(CANONICAL_STATUS_ORDER, (3, 3, 3, 2, 2, 2, 3, 2, 3), strict=True))
)


def renderer_fixture_to_jsonable(value: object) -> object:
    """Return plain JSON data for dataclass and Pydantic fixture roots."""
    if isinstance(value, BaseModel):
        return renderer_fixture_to_jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            row.name: renderer_fixture_to_jsonable(getattr(value, row.name))
            for row in fields(value)
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): renderer_fixture_to_jsonable(item)
            for key, item in mapping.items()
        }
    if isinstance(value, (tuple, list)):
        items = cast(tuple[object, ...] | list[object], value)
        return [renderer_fixture_to_jsonable(item) for item in items]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported renderer fixture value: {type(value).__name__}.")


_SYNTHETIC_DIGEST_A = sha256(b"synthetic-renderer-context").hexdigest()
_SYNTHETIC_DIGEST_B = sha256(b"synthetic-renderer-trajectory").hexdigest()
_SYNTHETIC_DIGEST_C = sha256(b"synthetic-renderer-canonical-bytes").hexdigest()


def _synthetic_replay_summary(
    episode_id: str,
    *,
    actor_pov: bool,
) -> ReplayArtifactSummaryV1:
    """Build path-free fixture provenance without claiming a persisted replay."""
    return ReplayArtifactSummaryV1(
        replay_reference=ReplayArtifactReferenceV1(
            artifact_id=f"{episode_id}:replay",
            episode_id=episode_id,
            context_digest_sha256=_SYNTHETIC_DIGEST_A,
            trajectory_content_digest_sha256=_SYNTHETIC_DIGEST_B,
            canonical_digest_sha256=_SYNTHETIC_DIGEST_C,
            canonical_byte_length=1,
        ),
        expected_transition_count=1,
        recorded_transition_count=1,
        recorded_frame_count=2,
        metric_report_availability=(
            ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1 if actor_pov else "missing"
        ),
    )


def _synthetic_researcher_presentation_pair(
    live_frame: ResearcherLiveDebuggerFrameV2,
) -> SyntheticFixturePresentationPair:
    """Package one exact synthetic researcher projection in both wire modes."""
    scene = live_frame.projection.scene
    events = live_frame.projection.incoming_events
    if scene.frame_index != 1 or events is None:
        raise ValueError(
            "synthetic researcher replay pairs require one incoming transition."
        )
    summary = _synthetic_replay_summary(scene.episode_id, actor_pov=False)
    completion = ReplayCompletionBadgeV1(
        episode_id=scene.episode_id,
        completion_state="complete",
        expected_transition_count=1,
        validated_transition_count=1,
        last_valid_frame_index=1,
        last_valid_frame_id=scene.frame_id,
        terminated=False,
        truncated=False,
        completion_bases=("declared_horizon",),
    )
    processing = ReplayProcessingBadgeV1(
        status="succeeded",
        processed_transition_count=1,
    )
    timeline_id = f"{summary.replay_reference.artifact_id}:timeline:researcher"
    replay_frame = ResearcherReplayViewerFrameV1(
        viewer_session_id=f"fixture-{live_frame.session_id}",
        revision=0,
        artifact_summary=summary,
        timeline_id=timeline_id,
        cursor=ReplayCursorV1(
            frame_index=1,
            final_frame_index=1,
            cursor_generation=1,
            choreography_generation=1,
        ),
        preset=live_frame.preset,
        verbose=live_frame.verbose,
        frame_id=scene.frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_transition_index=0,
        incoming_transition_id=scene.incoming_transition_id,
        completion=completion,
        processing=processing,
        show_ranges=live_frame.show_ranges,
        projection=live_frame.projection,
    )
    replay_timeline = ResearcherReplayTimelineV1(
        timeline_id=timeline_id,
        artifact_summary=summary,
        final_frame_index=1,
        completion=completion,
        rows=(
            ResearcherReplayTimelineRowV1(
                frame_index=0,
                frame_id=f"{scene.episode_id}:frame:0",
                simulator_step_count=scene.simulator_step_count - 1,
                incoming_transition_id=None,
                incoming_event_count=0,
            ),
            ResearcherReplayTimelineRowV1(
                frame_index=1,
                frame_id=scene.frame_id,
                simulator_step_count=scene.simulator_step_count,
                incoming_transition_id=scene.incoming_transition_id,
                incoming_event_count=len(events.events),
                endpoint_kind="declared_horizon",
            ),
        ),
    )
    return SyntheticFixturePresentationPair(
        audience="researcher",
        live_frame=live_frame,
        replay_frame=replay_frame,
        replay_timeline=replay_timeline,
    )


def _synthetic_pov_presentation_pair(
    live_frame: ActorPovLiveDebuggerFrameV2,
) -> SyntheticFixturePresentationPair:
    """Package one exact recipient projection in live and replay wire modes."""
    projection = live_frame.projection
    scene = projection.scene
    if scene.frame_index != 1 or projection.incoming_transition_id is None:
        raise ValueError("synthetic POV replay pairs require one incoming transition.")
    summary = _synthetic_replay_summary(scene.episode_id, actor_pov=True)
    completion = ActorPovReplayCompletionBadgeV1(
        episode_id=scene.episode_id,
        completion_state="complete",
        expected_transition_count=1,
        captured_transition_count=1,
        terminated=True,
        truncated=False,
        completion_bases=("task_terminal", "declared_horizon"),
        public_end_or_failure_reason="synthetic fixture complete",
    )
    public_agent_id = scene.self_actor.public_agent_id
    timeline_id = (
        f"{summary.replay_reference.artifact_id}:timeline:actor-pov:{public_agent_id}"
    )
    replay_frame = ActorPovReplayViewerFrameV1(
        viewer_session_id=f"fixture-{live_frame.session_id}",
        revision=0,
        artifact_summary=summary,
        timeline_id=timeline_id,
        cursor=ReplayCursorV1(
            frame_index=1,
            final_frame_index=1,
            cursor_generation=1,
            choreography_generation=1,
        ),
        preset=live_frame.preset,
        verbose=live_frame.verbose,
        pov_global_slot=scene.self_actor.global_slot,
        public_agent_id=public_agent_id,
        pov_frame_id=scene.pov_frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_pov_transition_id=projection.incoming_transition_id,
        completion=completion,
        processing_disclosure=ActorPovProcessingDisclosureV1(),
        projection=projection,
    )
    replay_timeline = ActorPovReplayTimelineV1(
        timeline_id=timeline_id,
        artifact_summary=summary,
        final_frame_index=1,
        pov_global_slot=scene.self_actor.global_slot,
        public_agent_id=public_agent_id,
        completion=completion,
        rows=(
            ActorPovReplayTimelineRowV1(
                frame_index=0,
                pov_frame_id=(
                    f"{scene.episode_id}:actor-pov:{public_agent_id}:frame:0"
                ),
                simulator_step_count=scene.simulator_step_count - 1,
                incoming_pov_transition_id=None,
                incoming_cue_count=0,
            ),
            ActorPovReplayTimelineRowV1(
                frame_index=1,
                pov_frame_id=scene.pov_frame_id,
                simulator_step_count=scene.simulator_step_count,
                incoming_pov_transition_id=projection.incoming_transition_id,
                incoming_cue_count=len(projection.incoming_cues),
                endpoint_kind="task_terminal_and_declared_horizon",
            ),
        ),
    )
    return SyntheticFixturePresentationPair(
        audience="agent_pov",
        live_frame=live_frame,
        replay_frame=replay_frame,
        replay_timeline=replay_timeline,
    )


def _class_mechanics() -> tuple[ClassMechanicsSceneV2, ...]:
    basic_modes: dict[int, BasicTargetModeV2] = {
        MAGE_CLASS_ID: "enemy",
        WARRIOR_CLASS_ID: "enemy",
        HUNTER_CLASS_ID: "enemy",
        ROGUE_CLASS_ID: "enemy",
        PRIEST_CLASS_ID: "ally",
    }
    ultimate_modes: dict[int, UltimateTargetModeV2] = {
        MAGE_CLASS_ID: "target_none",
        WARRIOR_CLASS_ID: "enemy",
        HUNTER_CLASS_ID: "enemy",
        ROGUE_CLASS_ID: "enemy",
        PRIEST_CLASS_ID: "ally",
    }
    status_rows_by_class: dict[int, list[ClassStatusMechanicSceneV2]] = {
        class_id: [] for class_id in range(1, 6)
    }
    for token_id in CANONICAL_STATUS_ORDER:
        definition = lookup_status_token(token_id)
        if definition.source_class_id is None:
            raise AssertionError(f"status {token_id!r} must have a source class")
        family, component, magnitude_kind, magnitude, breaks = _STATUS_DETAILS[token_id]
        status_rows_by_class[definition.source_class_id].append(
            ClassStatusMechanicSceneV2(
                status_channel=CATALOG_STATUS_ID_BY_CHANNEL.index(
                    _CATALOG_STATUS_ID_BY_TOKEN_ID[token_id]
                ),
                status_id=_CATALOG_STATUS_ID_BY_TOKEN_ID[token_id],
                family=family,
                source_action_component=component,
                duration_steps=_STATUS_DURATION_BY_TOKEN_ID[token_id],
                magnitude_kind=magnitude_kind,
                magnitude=magnitude,
                breaks_on_positive_damage=breaks,
            )
        )
    aura_rows_by_class: dict[int, tuple[ClassAuraMechanicSceneV2, ...]] = {
        class_id: () for class_id in range(1, 6)
    }
    aura_rows_by_class[MAGE_CLASS_ID] = (
        ClassAuraMechanicSceneV2(
            aura_id="mage_damage_amplification",
            radius=4.0,
            per_emitter_multiplier=1.1,
            stacking_rule="multiply_then_clamp",
            clamp_kind="ceiling",
            clamp_value=1.5,
        ),
    )
    aura_rows_by_class[WARRIOR_CLASS_ID] = (
        ClassAuraMechanicSceneV2(
            aura_id="warrior_damage_mitigation",
            radius=4.0,
            per_emitter_multiplier=0.9,
            stacking_rule="multiply_then_clamp",
            clamp_kind="floor",
            clamp_value=0.5,
        ),
    )
    return tuple(
        ClassMechanicsSceneV2(
            class_id=class_id,
            class_name=_CLASS_NAMES[class_id],
            maximum_health=100.0,
            body_radius=0.5,
            base_movement_speed=1.0,
            observation_radius=6.0,
            basic_target_mode=basic_modes[class_id],
            basic_interaction_radius=3.0,
            basic_raw_damage=0.0 if class_id == PRIEST_CLASS_ID else 10.0,
            basic_raw_healing=10.0 if class_id == PRIEST_CLASS_ID else 0.0,
            ultimate_target_mode=ultimate_modes[class_id],
            ultimate_interaction_radius=4.0,
            ultimate_cooldown_steps=5,
            ultimate_raw_damage=15.0 if class_id != PRIEST_CLASS_ID else 0.0,
            ultimate_raw_healing=20.0 if class_id == PRIEST_CLASS_ID else 0.0,
            out_of_combat_delay_steps=3,
            out_of_combat_health_regeneration_fraction_per_step=0.05,
            status_mechanics=tuple(
                sorted(
                    status_rows_by_class[class_id],
                    key=lambda status: status.status_channel,
                )
            ),
            aura_mechanics=aura_rows_by_class[class_id],
        )
        for class_id in range(1, 6)
    )


_CLASS_MECHANICS_V2 = _class_mechanics()


def _status(token_id: StatusTokenId, duration: int) -> StatusSceneV2:
    definition = lookup_status_token(token_id)
    if definition.source_class_id is None:
        raise AssertionError(f"status {token_id!r} must have a source class")
    family, component, magnitude_kind, magnitude, breaks = _STATUS_DETAILS[token_id]
    catalog_status_id = _CATALOG_STATUS_ID_BY_TOKEN_ID[token_id]
    return StatusSceneV2(
        status_channel=CATALOG_STATUS_ID_BY_CHANNEL.index(catalog_status_id),
        status_id=catalog_status_id,
        family=family,
        remaining_duration=duration,
        source_class_id=definition.source_class_id,
        source_class_name=_CLASS_NAMES[definition.source_class_id],
        source_action_component=component,
        magnitude_kind=magnitude_kind,
        magnitude=magnitude,
        breaks_on_positive_damage=breaks,
    )


def _statuses(token_ids: tuple[StatusTokenId, ...]) -> tuple[StatusSceneV2, ...]:
    return tuple(
        _status(token_id, _STATUS_DURATION_BY_TOKEN_ID[token_id])
        for token_id in sorted(token_ids, key=CANONICAL_STATUS_ORDER.index)
    )


def _agents(
    positions: tuple[tuple[float, float], ...],
    *,
    status_tokens: Mapping[int, tuple[StatusTokenId, ...]] | None = None,
    health: Mapping[int, float] | None = None,
    modifier_values: Mapping[int, tuple[float, float]] | None = None,
    cooldowns: Mapping[int, int] | None = None,
    out_of_combat: Mapping[int, int] | None = None,
    class_ids: tuple[int, ...] = _CLASS_IDS,
    included_slots: tuple[int, ...] = tuple(range(10)),
    corpses: tuple[int, ...] = (),
    spawn_shields: Mapping[int, int] | None = None,
    respawn_event_ids: Mapping[int, str] | None = None,
) -> tuple[AgentSceneV2, ...]:
    if len(positions) != 10 or len(class_ids) != 10:
        raise ValueError("synthetic position/class tables require ten rows.")
    status_tokens = status_tokens or {}
    health = health or {}
    modifier_values = modifier_values or {}
    cooldowns = cooldowns or {}
    out_of_combat = out_of_combat or {}
    spawn_shields = spawn_shields or {}
    respawn_event_ids = respawn_event_ids or {}
    rows: list[AgentSceneV2] = []
    for slot in included_slots:
        mage_modifier, warrior_modifier = modifier_values.get(slot, (1.0, 1.0))
        respawn_event_id = respawn_event_ids.get(slot)
        rows.append(
            AgentSceneV2(
                global_slot=slot,
                public_agent_id=str(slot),
                team_id=_TEAM_IDS[slot],
                team_local_slot=slot % 5,
                class_id=class_ids[slot],
                position=positions[slot],
                radius=0.5,
                life_state="corpse" if slot in corpses else "alive",
                current_health=health.get(slot, 0.0 if slot in corpses else 100.0),
                max_health=100.0,
                effective_movement_speed=0.0 if slot in corpses else 1.0,
                ultimate_cooldown_remaining=cooldowns.get(slot, 0),
                spawn_shield_remaining=spawn_shields.get(slot, 0),
                steps_until_out_of_combat=out_of_combat.get(slot, 0),
                respawned_on_incoming_transition=respawn_event_id is not None,
                respawn_event_id=respawn_event_id,
                statuses=_statuses(status_tokens.get(slot, ())),
                aura_modifiers=(
                    AuraRecipientModifierSceneV2(
                        aura_id="mage_damage_amplification",
                        multiplier=mage_modifier,
                    ),
                    AuraRecipientModifierSceneV2(
                        aura_id="warrior_damage_mitigation",
                        multiplier=warrior_modifier,
                    ),
                ),
            )
        )
    return tuple(rows)


def _aura_field(
    agents: Mapping[int, AgentSceneV2],
    source_global_slot: int,
    aura_id: AuraIdV2,
) -> AuraFieldSceneV2:
    source = agents[source_global_slot]
    class_mechanics = _CLASS_MECHANICS_V2[source.class_id - 1]
    mechanic = next(
        (row for row in class_mechanics.aura_mechanics if row.aura_id == aura_id),
        None,
    )
    if mechanic is None:
        raise ValueError("synthetic aura field must join its source class mechanic.")
    return AuraFieldSceneV2(
        aura_id=aura_id,
        source_global_slot=source_global_slot,
        source_public_agent_id=source.public_agent_id,
        source_class_id=source.class_id,
        source_class_name=_CLASS_NAMES[source.class_id],
        source_alive=source.life_state == "alive",
        center=source.position,
        radius=mechanic.radius,
        beneficiary_relation="same_team",
        per_emitter_multiplier=mechanic.per_emitter_multiplier,
        stacking_rule=mechanic.stacking_rule,
        clamp_kind=mechanic.clamp_kind,
        clamp_value=mechanic.clamp_value,
    )


def _anchor(
    agents: Mapping[int, AgentSceneV2],
    positions: Mapping[int, tuple[float, float]],
    slot: int,
    phase: Literal["transition_start", "post_charge", "successor"],
) -> VisualAgentAnchorV2:
    return VisualAgentAnchorV2(
        phase=phase,
        global_slot=slot,
        public_agent_id=agents[slot].public_agent_id,
        position=positions[slot],
    )


class _EventIdentity(TypedDict):
    event_id: str
    transition_id: str
    ordinal: int


class _StatusEventArguments(_EventIdentity):
    recipient_global_slot: int
    status_channel: int
    status_id: str
    recipient_anchor: VisualAgentAnchorV2


def _event_from_spec(
    *,
    event_id: str,
    transition_id: str,
    ordinal: int,
    spec: Mapping[str, object],
    agents: Mapping[int, AgentSceneV2],
    start: Mapping[int, tuple[float, float]],
    post_charge: Mapping[int, tuple[float, float]],
    successor: Mapping[int, tuple[float, float]],
) -> VisualEventV2:
    event_type = cast(str, spec["event_type"])
    common: _EventIdentity = {
        "event_id": event_id,
        "transition_id": transition_id,
        "ordinal": ordinal,
    }
    if event_type == "action_rejected":
        slot = cast(int, spec["actor"])
        active = cast(bool, spec.get("active", True))
        return ActionRejectedEventV2(
            **common,
            actor_global_slot=slot,
            actor_public_agent_id=str(slot),
            actor_configured_active=active,
            rejection_component=cast(
                Literal["domain", "movement", "combat_pair"],
                spec.get("component", "movement"),
            ),
            submitted_move_action=cast(int, spec.get("move_action", 1)),
            submitted_select_target_action=cast(int, spec.get("target_action", 1)),
            submitted_use_ultimate_action=cast(int, spec.get("ultimate_action", 0)),
            actor_anchor=(
                _anchor(agents, start, slot, "transition_start") if active else None
            ),
        )
    if event_type == "ability_activated":
        source = cast(int, spec["source"])
        recipient = cast(int | None, spec.get("recipient"))
        return AbilityActivatedEventV2(
            **common,
            source_global_slot=source,
            ability_component=cast(Literal["basic", "ultimate"], spec["component"]),
            recipient_global_slot=recipient,
            source_anchor=_anchor(agents, start, source, "transition_start"),
            recipient_anchor=(
                None
                if recipient is None
                else _anchor(agents, start, recipient, "transition_start")
            ),
        )
    if event_type == "source_damage_output":
        source = cast(int, spec["source"])
        recipient = cast(int | None, spec.get("recipient"))
        return SourceDamageOutputEventV2(
            **common,
            source_global_slot=source,
            recipient_global_slot=recipient,
            raw_damage_output=cast(float, spec.get("raw", 10.0)),
            source_modified_damage_output=cast(float, spec.get("modified", 12.0)),
            recipient_damage_modifier=cast(float, spec.get("modifier", 0.8)),
            mage_damage_aura_covering_emitter_global_slots=cast(
                tuple[int, ...], spec.get("mage_emitters", ())
            ),
            warrior_mitigation_aura_covering_emitter_global_slots=cast(
                tuple[int, ...], spec.get("warrior_emitters", ())
            ),
            source_anchor=_anchor(agents, start, source, "transition_start"),
            recipient_anchor=(
                None
                if recipient is None
                else _anchor(agents, start, recipient, "transition_start")
            ),
        )
    if event_type == "source_healing_output":
        source = cast(int, spec["source"])
        recipient = cast(int | None, spec.get("recipient"))
        return SourceHealingOutputEventV2(
            **common,
            source_global_slot=source,
            recipient_global_slot=recipient,
            raw_healing_output=cast(float, spec.get("raw", 10.0)),
            source_modified_healing_output=cast(float, spec.get("modified", 10.0)),
            recipient_healing_modifier=cast(float, spec.get("modifier", 1.0)),
            source_anchor=_anchor(agents, start, source, "transition_start"),
            recipient_anchor=(
                None
                if recipient is None
                else _anchor(agents, start, recipient, "transition_start")
            ),
        )
    if event_type == "recipient_health_resolution":
        recipient = cast(int, spec["recipient"])
        return RecipientHealthResolutionEventV2(
            **common,
            recipient_global_slot=recipient,
            transition_start_health=cast(float, spec["before"]),
            total_effective_damage=cast(float, spec.get("damage", 0.0)),
            total_effective_healing=cast(float, spec.get("healing", 0.0)),
            health_after_combat_resolution=cast(float, spec["after"]),
            realized_net_health_change=cast(float, spec["delta"]),
            recipient_anchor=_anchor(agents, start, recipient, "transition_start"),
        )
    if event_type in {
        "combat_countdown_reset",
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
    }:
        slot = cast(int, spec["agent"])
        agent_anchor = _anchor(agents, start, slot, "transition_start")
        if event_type == "combat_countdown_reset":
            return CombatCountdownResetEventV2(
                **common, agent_global_slot=slot, agent_anchor=agent_anchor
            )
        if event_type == "health_regenerated":
            return HealthRegeneratedEventV2(
                **common,
                agent_global_slot=slot,
                actual_health_regenerated=cast(float, spec.get("amount", 2.0)),
                agent_anchor=agent_anchor,
            )
        if event_type == "cooldown_started":
            return CooldownStartedEventV2(
                **common, agent_global_slot=slot, agent_anchor=agent_anchor
            )
        return CooldownReadyEventV2(
            **common, agent_global_slot=slot, agent_anchor=agent_anchor
        )
    if event_type in {
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
    }:
        slot = cast(int, spec["agent"])
        if event_type == "charge_phase_displacement":
            phase_start, phase_end = start, post_charge
            event_class = ChargePhaseDisplacementEventV2
            start_phase: VisualAnchorPhaseV2 = "transition_start"
            end_phase: VisualAnchorPhaseV2 = "post_charge"
        else:
            phase_start, phase_end = post_charge, successor
            event_class = OrdinaryMovementPhaseDisplacementEventV2
            start_phase = "post_charge"
            end_phase = "successor"
        displacement = (
            phase_end[slot][0] - phase_start[slot][0],
            phase_end[slot][1] - phase_start[slot][1],
        )
        return event_class(
            **common,
            agent_global_slot=slot,
            realized_displacement=displacement,
            start_anchor=_anchor(agents, phase_start, slot, start_phase),
            end_anchor=_anchor(agents, phase_end, slot, end_phase),
        )
    if event_type == "agent_died":
        recipient = cast(int, spec["recipient"])
        return AgentDiedEventV2(
            **common,
            recipient_global_slot=recipient,
            recipient_anchor=_anchor(agents, successor, recipient, "successor"),
        )
    if event_type == "lethal_damage_contribution":
        source = cast(int, spec["source"])
        recipient = cast(int, spec["recipient"])
        return LethalDamageContributionEventV2(
            **common,
            source_global_slot=source,
            recipient_global_slot=recipient,
            attributed_death_damage=cast(float, spec.get("amount", 5.0)),
            source_anchor=_anchor(agents, successor, source, "successor"),
            recipient_anchor=_anchor(agents, successor, recipient, "successor"),
        )
    if event_type in {
        "status_aged_to_zero",
        "status_broken_by_damage",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
    }:
        recipient = cast(int, spec["recipient"])
        catalog_status_id = cast(CatalogStatusId, spec["status_id"])
        status_common: _StatusEventArguments = {
            **common,
            "recipient_global_slot": recipient,
            "status_channel": CATALOG_STATUS_ID_BY_CHANNEL.index(catalog_status_id),
            "status_id": catalog_status_id,
            "recipient_anchor": _anchor(agents, successor, recipient, "successor"),
        }
        if event_type == "status_aged_to_zero":
            return StatusAgedToZeroEventV2(**status_common)
        if event_type == "status_broken_by_damage":
            return StatusBrokenByDamageEventV2(**status_common)
        if event_type == "status_refreshed_or_extended":
            return StatusRefreshedOrExtendedEventV2(**status_common)
        if event_type == "status_cleared_by_new_death":
            return StatusClearedByNewDeathEventV2(**status_common)
        source = cast(int, spec["source"])
        return StatusAppliedEventV2(
            **status_common,
            source_global_slot=source,
            source_anchor=_anchor(agents, successor, source, "successor"),
        )
    if event_type == "spawn_shield_expired":
        slot = cast(int, spec["agent"])
        return SpawnShieldExpiredEventV2(
            **common,
            agent_global_slot=slot,
            agent_anchor=_anchor(agents, successor, slot, "successor"),
        )
    if event_type == "respawn_wave_occurred":
        team_index = cast(int, spec["team_index"])
        return RespawnWaveOccurredEventV2(
            **common,
            team_index=team_index,
            team_id=team_index + 1,
            team_anchor=VisualTeamAnchorV2(
                phase="successor",
                team_index=team_index,
                team_id=team_index + 1,
            ),
        )
    if event_type == "agent_respawned":
        slot = cast(int, spec["agent"])
        return AgentRespawnedEventV2(
            **common,
            agent_global_slot=slot,
            team_id=agents[slot].team_id,
            realized_successor_position=successor[slot],
            agent_anchor=_anchor(agents, successor, slot, "successor"),
        )
    if event_type == "team_deathmatch_score_changed":
        team_index = cast(Literal[0, 1], spec["team_index"])
        team_id = team_index + 1
        previous_score = cast(int, spec.get("previous_score", 0))
        score_increment = cast(int, spec.get("score_increment", 1))
        return TeamDeathmatchScoreChangedEventV2(
            **common,
            team_index=team_index,
            team_id=team_id,
            score_increment=score_increment,
            previous_score=previous_score,
            successor_score=previous_score + score_increment,
            team_anchor=VisualTeamAnchorV2(
                phase="successor",
                team_index=team_index,
                team_id=team_id,
            ),
        )
    if event_type == "team_deathmatch_completed":
        return TeamDeathmatchCompletedEventV2(
            **common,
            outcome=cast(
                Literal["team_a_win", "team_b_win", "draw"],
                spec.get("outcome", "team_a_win"),
            ),
            completion_basis=cast(
                Literal[
                    "score_threshold",
                    "horizon",
                    "score_threshold_at_horizon",
                ],
                spec.get("completion_basis", "score_threshold"),
            ),
        )
    raise AssertionError(f"unsupported synthetic V2 event: {event_type}.")


def _batch(
    name: str,
    agents: tuple[AgentSceneV2, ...],
    specs: tuple[Mapping[str, object], ...],
    *,
    start_positions: Mapping[int, tuple[float, float]] | None = None,
    post_charge_positions: Mapping[int, tuple[float, float]] | None = None,
) -> VisualEventBatchV2:
    episode_id = f"synthetic:{name}"
    transition_id = f"{episode_id}:transition:0"
    by_slot = {agent.global_slot: agent for agent in agents}
    successor = {slot: agent.position for slot, agent in by_slot.items()}
    start = {**successor, **(start_positions or {})}
    post_charge = {**successor, **(post_charge_positions or {})}
    trajectories = tuple(
        VisualAgentPhaseTrajectoryV2(
            global_slot=slot,
            public_agent_id=agent.public_agent_id,
            transition_start=_anchor(by_slot, start, slot, "transition_start"),
            post_charge=_anchor(by_slot, post_charge, slot, "post_charge"),
            successor=_anchor(by_slot, successor, slot, "successor"),
        )
        for slot, agent in sorted(by_slot.items())
    )
    events = tuple(
        _event_from_spec(
            event_id=f"{transition_id}:event:{ordinal:04d}",
            transition_id=transition_id,
            ordinal=ordinal,
            spec=spec,
            agents=by_slot,
            start=start,
            post_charge=post_charge,
            successor=successor,
        )
        for ordinal, spec in enumerate(specs)
    )
    active_slots = set(by_slot)
    return VisualEventBatchV2(
        schema_version=EVENT_V2_SCHEMA_VERSION,
        episode_id=episode_id,
        transition_index=0,
        transition_id=transition_id,
        start_frame_id=f"{episode_id}:frame:0",
        successor_frame_id=f"{episode_id}:frame:1",
        start_simulator_step_count=0,
        successor_simulator_step_count=1,
        public_agent_id_by_global_slot=tuple(str(slot) for slot in range(10)),
        configured_active_by_global_slot=tuple(
            slot in active_slots for slot in range(10)
        ),
        agent_phase_trajectories=trajectories,
        events=events,
    )


def _scene(
    name: str,
    agents: tuple[AgentSceneV2, ...],
    *,
    event_batch: VisualEventBatchV2 | None = None,
    map_scene: MapSceneV1,
    aura_fields: tuple[AuraFieldSceneV2, ...] = (),
    ranges: tuple[RangeSceneV1, ...] = (),
    selection: SelectionSceneV1 | None = None,
    selected_legality: SelectedLegalitySceneV1 | None = None,
    visibility_by_slot: Mapping[int, bool] | None = None,
    respawn_wave_countdowns: tuple[int, int] = (4, 7),
    class_mechanics: tuple[ClassMechanicsSceneV2, ...] = _CLASS_MECHANICS_V2,
    badge: str = "PRIVILEGED RESEARCHER VIEW · SYNTHETIC FIXTURE",
) -> BattlefieldSceneV2:
    episode_id = f"synthetic:{name}"
    frame_index = 0 if event_batch is None else 1
    if event_batch is not None and event_batch.episode_id != episode_id:
        raise ValueError("synthetic scene and event batch names must join.")
    resolved_visibility = (
        {}
        if selection is None
        else {
            agent.global_slot: (
                visibility_by_slot[agent.global_slot]
                if visibility_by_slot is not None
                else agent.global_slot < 7
            )
            for agent in agents
        }
    )
    return BattlefieldSceneV2(
        schema_version=SCENE_V2_SCHEMA_VERSION,
        audience="researcher",
        audience_badge=badge,
        episode_id=episode_id,
        frame_index=frame_index,
        frame_id=f"{episode_id}:frame:{frame_index}",
        simulator_step_count=frame_index,
        incoming_transition_id=(
            None if event_batch is None else event_batch.transition_id
        ),
        incoming_event_ids=(
            ()
            if event_batch is None
            else tuple(event.event_id for event in event_batch.events)
        ),
        map=map_scene,
        agents=agents,
        aura_fields=aura_fields,
        class_mechanics=class_mechanics,
        spawn_pads=tuple(
            SpawnPadSceneV2(
                team_id=agent.team_id,
                team_local_slot=agent.team_local_slot,
                assigned_global_slot=agent.global_slot,
                assigned_public_agent_id=agent.public_agent_id,
                position=agent.position,
            )
            for agent in agents
        ),
        respawn_waves=(
            RespawnWaveSceneV2(
                team_index=0,
                team_id=1,
                period_steps=10,
                countdown_steps=respawn_wave_countdowns[0],
            ),
            RespawnWaveSceneV2(
                team_index=1,
                team_id=2,
                period_steps=10,
                countdown_steps=respawn_wave_countdowns[1],
            ),
        ),
        ranges=ranges,
        selection=selection,
        next_decision_selected_legality=selected_legality,
        observer_visibility=tuple(
            ObserverVisibilitySceneV1(
                observer_global_slot=selection.controlled_global_slot,
                candidate_global_slot=agent.global_slot,
                visible=resolved_visibility[agent.global_slot],
            )
            for agent in agents
            if selection is not None
        ),
    )


def _activation_specs(
    rows: tuple[tuple[str, int, int | None], ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        MappingProxyType(
            {
                "event_type": "ability_activated",
                "component": (
                    "basic" if token in ("basic_damage", "basic_heal") else "ultimate"
                ),
                "source": source,
                "recipient": recipient,
            }
        )
        for token, source, recipient in rows
    )


_CROWDED_POSITIONS = (
    (2.5, 3.4),
    (5.25, 3.4),
    (8.0, 3.4),
    (10.75, 3.4),
    (13.5, 3.4),
    (2.5, 8.6),
    (5.25, 8.6),
    (8.0, 8.6),
    (10.75, 8.6),
    (13.5, 8.6),
)
_CROWDED_PRE_ANCHORS = MappingProxyType(
    {
        0: (2.5, 5.0),
        1: (5.25, 4.0),
        2: (8.0, 5.0),
        3: (10.75, 5.0),
        4: (13.5, 5.0),
        5: (2.5, 7.0),
        6: (5.25, 8.0),
        7: (8.0, 7.0),
        8: (10.75, 7.0),
        9: (13.5, 7.0),
    }
)
_CROWDED_STATUS_TOKENS = MappingProxyType(
    {slot: CANONICAL_STATUS_ORDER for slot in range(10)}
)
_CROWDED_MODIFIERS = MappingProxyType({slot: (1.2, 0.8) for slot in range(10)})
_CROWDED_HEALTH = MappingProxyType(
    {
        0: 82.0,
        1: 88.0,
        2: 100.0,
        3: 92.0,
        4: 80.0,
        5: 82.0,
        6: 88.0,
        7: 100.0,
        8: 92.0,
        9: 80.0,
    }
)
_CROWDED_AGENTS = _agents(
    _CROWDED_POSITIONS,
    status_tokens=_CROWDED_STATUS_TOKENS,
    health=_CROWDED_HEALTH,
    modifier_values=_CROWDED_MODIFIERS,
)
_CROWDED_AGENT_MAP = {agent.global_slot: agent for agent in _CROWDED_AGENTS}
_CROWDED_ACTIVATION_ROWS = (
    ("basic_damage", 0, 5),
    ("basic_damage", 5, 0),
    ("warrior_charge", 1, 6),
    ("warrior_charge", 6, 1),
    ("hunter_trap", 2, 7),
    ("hunter_trap", 7, 2),
    ("rogue_poison", 3, 8),
    ("rogue_poison", 8, 3),
    ("holy_word", 4, 4),
    ("holy_word", 9, 9),
)
_CROWDED_HEALTH_ROWS = (
    (0, 90.0, 82.0),
    (1, 100.0, 88.0),
    (3, 100.0, 92.0),
    (4, 70.0, 80.0),
    (5, 90.0, 82.0),
    (6, 100.0, 88.0),
    (8, 100.0, 92.0),
    (9, 70.0, 80.0),
)
_CROWDED_STATUS_ROWS = (
    (6, "warrior_charge_stun", 1),
    (6, "warrior_charge_slow", 1),
    (1, "warrior_charge_stun", 6),
    (1, "warrior_charge_slow", 6),
    (7, "hunter_trap_stun", 2),
    (2, "hunter_trap_stun", 7),
    (8, "rogue_poison_stun", 3),
    (8, "rogue_poison_slow", 3),
    (8, "rogue_poison_anti_heal", 3),
    (3, "rogue_poison_stun", 8),
    (3, "rogue_poison_slow", 8),
    (3, "rogue_poison_anti_heal", 8),
)
_CROWDED_SPECS = (
    *_activation_specs(_CROWDED_ACTIVATION_ROWS),
    *(
        MappingProxyType(
            {
                "event_type": "recipient_health_resolution",
                "recipient": slot,
                "before": before,
                "after": after,
                "delta": after - before,
                "damage": max(before - after, 0.0),
                "healing": max(after - before, 0.0),
            }
        )
        for slot, before, after in _CROWDED_HEALTH_ROWS
    ),
    MappingProxyType({"event_type": "charge_phase_displacement", "agent": 1}),
    MappingProxyType({"event_type": "charge_phase_displacement", "agent": 6}),
    *(
        MappingProxyType(
            {
                "event_type": "status_applied",
                "recipient": recipient,
                "status_id": status_id,
                "source": source,
            }
        )
        for recipient, status_id, source in _CROWDED_STATUS_ROWS
    ),
)
_CROWDED_BATCH = _batch(
    "crowded_teamfight",
    _CROWDED_AGENTS,
    _CROWDED_SPECS,
    start_positions=_CROWDED_PRE_ANCHORS,
)
_CROWDED_SCENE = _scene(
    "crowded_teamfight",
    _CROWDED_AGENTS,
    event_batch=_CROWDED_BATCH,
    map_scene=MapSceneV1(
        width=16.0,
        height=12.0,
        obstacles=(
            ObstacleSceneV1(
                obstacle_id="synthetic-crowded-pillar",
                kind="pillar",
                center=(1.25, 6.0),
                radius=0.7,
            ),
            ObstacleSceneV1(
                obstacle_id="synthetic-crowded-wall",
                kind="wall",
                center=(14.75, 6.0),
                width=0.8,
                height=4.0,
                theta=0.0,
            ),
        ),
    ),
    aura_fields=(
        _aura_field(_CROWDED_AGENT_MAP, 0, "mage_damage_amplification"),
        _aura_field(_CROWDED_AGENT_MAP, 1, "warrior_damage_mitigation"),
        _aura_field(_CROWDED_AGENT_MAP, 5, "mage_damage_amplification"),
        _aura_field(_CROWDED_AGENT_MAP, 6, "warrior_damage_mitigation"),
    ),
    ranges=(
        RangeSceneV1(
            global_slot=0, center=_CROWDED_POSITIONS[0], radius=6.0, kind="observation"
        ),
        RangeSceneV1(
            global_slot=0, center=_CROWDED_POSITIONS[0], radius=3.0, kind="basic"
        ),
        RangeSceneV1(
            global_slot=0, center=_CROWDED_POSITIONS[0], radius=4.0, kind="ultimate"
        ),
    ),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=7),
    selected_legality=SelectedLegalitySceneV1(
        controlled_global_slot=0,
        target_global_slot=7,
        target_action=8,
        lane_0_available=True,
        lane_1_available=False,
        armed_lane=1,
        armed_pair_legal=False,
    ),
)


_REQUIRED_DOCK_POSITIONS = (
    (5.8, 6.65),
    (6.9, 6.65),
    (8.0, 6.65),
    (9.1, 6.65),
    (10.2, 6.65),
    (5.8, 5.35),
    (6.9, 5.35),
    (10.2, 5.35),
    (8.0, 5.35),
    (9.1, 5.35),
)
_REQUIRED_DOCK_AGENTS = _agents(
    _REQUIRED_DOCK_POSITIONS,
    status_tokens=_CROWDED_STATUS_TOKENS,
    cooldowns={slot: 30 for slot in range(10)},
)
_REQUIRED_DOCK_CLASS_MECHANICS = tuple(
    replace(mechanics, ultimate_cooldown_steps=30) for mechanics in _CLASS_MECHANICS_V2
)
_REQUIRED_DOCK_SCENE = _scene(
    "required_dock_fallback",
    _REQUIRED_DOCK_AGENTS,
    map_scene=MapSceneV1(width=16.0, height=12.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=7),
    class_mechanics=_REQUIRED_DOCK_CLASS_MECHANICS,
)


_ROUTE_POSITIONS = (
    (1.5, 1.5),
    (1.5, 3.5),
    (1.5, 5.5),
    (1.5, 5.8),
    (6.0, 8.0),
    (10.5, 1.5),
    (10.5, 3.5),
    (10.5, 5.5),
    (10.5, 5.8),
    (6.08, 8.04),
)
_ROUTE_AGENTS = _agents(_ROUTE_POSITIONS, class_ids=(HUNTER_CLASS_ID,) * 10)
_ROUTE_ROWS = (
    ("basic_damage", 0, 5),
    ("basic_damage", 5, 0),
    ("basic_damage", 1, 6),
    ("basic_damage", 1, 6),
    ("basic_damage", 2, 7),
    ("basic_damage", 3, 8),
    ("basic_damage", 0, 8),
    ("basic_damage", 3, 5),
    ("basic_damage", 4, 9),
)
_ROUTE_START_POSITIONS = MappingProxyType(
    {
        0: (3.0, 3.0),
        1: (3.0, 4.0),
        2: (3.0, 5.0),
        3: (3.0, 5.2),
        4: (4.2, 6.0),
        5: (5.0, 3.0),
        6: (5.0, 4.0),
        7: (5.0, 5.0),
        8: (5.0, 5.2),
        9: (4.28, 6.04),
    }
)
_ROUTE_BATCH = _batch(
    "route_collision",
    _ROUTE_AGENTS,
    _activation_specs(_ROUTE_ROWS),
    start_positions=_ROUTE_START_POSITIONS,
)
_ROUTE_SCENE = _scene(
    "route_collision",
    _ROUTE_AGENTS,
    event_batch=_ROUTE_BATCH,
    map_scene=MapSceneV1(width=12.0, height=10.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
)


_MIXED_POSITIONS = (
    (3.0, 4.0),
    (2.0, 2.0),
    (2.0, 6.0),
    (2.0, 8.0),
    (2.0, 10.0),
    (7.0, 4.0),
    (8.0, 2.0),
    (8.0, 6.0),
    (8.0, 8.0),
    (7.0, 5.5),
)
_MIXED_AGENTS = _agents(_MIXED_POSITIONS, health={5: 50.0}, included_slots=(0, 5, 9))
_MIXED_SPECS = (
    *_activation_specs((("basic_damage", 0, 5), ("basic_heal", 9, 5))),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 5,
            "before": 50.0,
            "after": 50.0,
            "delta": 0.0,
            "damage": 10.0,
            "healing": 10.0,
        }
    ),
)
_MIXED_BATCH = _batch(
    "mixed_net_zero",
    _MIXED_AGENTS,
    _MIXED_SPECS,
    start_positions={0: (4.0, 4.0)},
)
_MIXED_SCENE = _scene(
    "mixed_net_zero",
    _MIXED_AGENTS,
    event_batch=_MIXED_BATCH,
    map_scene=MapSceneV1(width=10.0, height=12.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
)


_VOCABULARY_POSITIONS = (
    (2.0, 3.0),
    (5.0, 3.0),
    (8.0, 3.0),
    (11.0, 3.0),
    (14.0, 3.0),
    (2.0, 9.0),
    (5.0, 9.0),
    (8.0, 9.0),
    (11.0, 9.0),
    (14.0, 9.0),
)
_VOCABULARY_AGENTS = _agents(
    _VOCABULARY_POSITIONS,
    status_tokens={
        0: ("stun_warrior_charge", "stun_hunter_trap", "stun_rogue_poison"),
        2: ("slow_warrior_charge", "slow_hunter_basic", "slow_rogue_poison"),
    },
    health={5: 87.654, 9: 78.5},
    cooldowns={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
)
_VOCABULARY_AGENT_MAP = {agent.global_slot: agent for agent in _VOCABULARY_AGENTS}
_VOCABULARY_ROWS = (
    ("basic_damage", 0, 5),
    ("basic_damage", 1, 6),
    ("basic_damage", 2, 7),
    ("basic_damage", 3, 8),
    ("basic_heal", 4, 4),
    ("mage_burst", 0, None),
    ("warrior_charge", 1, 6),
    ("hunter_trap", 2, 7),
    ("rogue_poison", 3, 8),
    ("holy_word", 4, 4),
)
_VOCABULARY_SPECS = (
    *_activation_specs(_VOCABULARY_ROWS),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 5,
            "before": 99.999,
            "after": 87.654,
            "delta": -12.345,
            "damage": 12.345,
            "healing": 0.0,
        }
    ),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 9,
            "before": 70.0,
            "after": 78.5,
            "delta": 8.5,
            "damage": 0.0,
            "healing": 8.5,
        }
    ),
)
_VOCABULARY_START_POSITIONS = MappingProxyType(
    {
        0: (2.0, 5.0),
        1: (5.0, 5.0),
        2: (8.0, 5.0),
        3: (11.0, 5.0),
        4: (14.0, 5.0),
        5: (2.0, 7.0),
        6: (5.0, 7.0),
        7: (8.0, 7.0),
        8: (11.0, 7.0),
        9: (14.0, 7.0),
    }
)
_VOCABULARY_BATCH = _batch(
    "visual_vocabulary",
    _VOCABULARY_AGENTS,
    _VOCABULARY_SPECS,
    start_positions=_VOCABULARY_START_POSITIONS,
)
_VOCABULARY_SCENE = _scene(
    "visual_vocabulary",
    _VOCABULARY_AGENTS,
    event_batch=_VOCABULARY_BATCH,
    map_scene=MapSceneV1(width=16.0, height=12.0),
    aura_fields=(
        _aura_field(_VOCABULARY_AGENT_MAP, 0, "mage_damage_amplification"),
        _aura_field(_VOCABULARY_AGENT_MAP, 1, "warrior_damage_mitigation"),
    ),
    ranges=(
        RangeSceneV1(
            global_slot=0,
            kind="observation",
            center=_VOCABULARY_POSITIONS[0],
            radius=_CLASS_MECHANICS_V2[
                _VOCABULARY_AGENTS[0].class_id - 1
            ].observation_radius,
        ),
        *(
            RangeSceneV1(
                global_slot=slot,
                kind="basic",
                center=_VOCABULARY_POSITIONS[slot],
                radius=_CLASS_MECHANICS_V2[
                    _VOCABULARY_AGENTS[slot].class_id - 1
                ].basic_interaction_radius,
            )
            for slot in range(5)
        ),
        RangeSceneV1(
            global_slot=0,
            kind="ultimate",
            center=_VOCABULARY_POSITIONS[0],
            radius=_CLASS_MECHANICS_V2[
                _VOCABULARY_AGENTS[0].class_id - 1
            ].ultimate_interaction_radius,
        ),
    ),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
    badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC VISUAL VOCABULARY",
)


_DURABLE_POSITIONS = (
    (4.0, 5.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (12.0, 5.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
)
_DURABLE_AGENTS = _agents(
    _DURABLE_POSITIONS,
    status_tokens={
        0: ("stun_warrior_charge", "stun_hunter_trap", "stun_rogue_poison"),
        5: ("slow_warrior_charge", "slow_hunter_basic", "slow_rogue_poison"),
    },
    included_slots=(0, 5),
)
_DURABLE_SCENE = _scene(
    "durable_controls",
    _DURABLE_AGENTS,
    map_scene=MapSceneV1(width=16.0, height=10.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
    badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC DURABLE CONTROLS",
)


_VIEWPORTS = (
    ViewportCaseV1(label="desktop", width=1440, height=900, expected_layout="split"),
    ViewportCaseV1(label="compact", width=1024, height=768, expected_layout="split"),
    ViewportCaseV1(label="minimum", width=960, height=600, expected_layout="split"),
    ViewportCaseV1(label="stacked", width=800, height=900, expected_layout="stacked"),
)
_VIEWPORT_SPECS = (
    *_activation_specs((("basic_damage", 0, 5), ("hunter_trap", 2, 7))),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 0,
            "before": 90.0,
            "after": 82.0,
            "delta": -8.0,
            "damage": 8.0,
            "healing": 0.0,
        }
    ),
)
_VIEWPORT_BATCH = _batch(
    "viewport_matrix",
    _CROWDED_AGENTS,
    _VIEWPORT_SPECS,
    start_positions=_CROWDED_PRE_ANCHORS,
)
_VIEWPORT_AURA_SPECS: tuple[tuple[int, AuraIdV2], ...] = (
    (0, "mage_damage_amplification"),
    (1, "warrior_damage_mitigation"),
    (5, "mage_damage_amplification"),
    (6, "warrior_damage_mitigation"),
)
_VIEWPORT_SCENE = _scene(
    "viewport_matrix",
    _CROWDED_AGENTS,
    event_batch=_VIEWPORT_BATCH,
    map_scene=_CROWDED_SCENE.map,
    aura_fields=tuple(
        _aura_field(_CROWDED_AGENT_MAP, slot, aura_id)
        for slot, aura_id in _VIEWPORT_AURA_SPECS
    ),
    ranges=_CROWDED_SCENE.ranges,
    selection=_CROWDED_SCENE.selection,
    selected_legality=_CROWDED_SCENE.next_decision_selected_legality,
)


_GRAMMAR_NAME = "canonical_event_vocabulary"
_GRAMMAR_EPISODE = f"synthetic:{_GRAMMAR_NAME}"
_GRAMMAR_TRANSITION = f"{_GRAMMAR_EPISODE}:transition:0"
_GRAMMAR_POSITIONS = (
    _VOCABULARY_POSITIONS[0],
    (5.0, 6.0),
    _VOCABULARY_POSITIONS[2],
    _VOCABULARY_POSITIONS[3],
    (2.0, 6.0),
    *_VOCABULARY_POSITIONS[5:],
)
_GRAMMAR_AGENTS = _agents(
    _GRAMMAR_POSITIONS,
    status_tokens={
        0: ("priest_freedom",),
        8: ("slow_rogue_poison",),
    },
    health={6: 0.0},
    cooldowns={1: 5},
    out_of_combat={1: 3},
    corpses=(6,),
    spawn_shields={9: 3},
    respawn_event_ids={9: f"{_GRAMMAR_TRANSITION}:event:0022"},
)
_GRAMMAR_SPECS = tuple(
    MappingProxyType(row)
    for row in (
        {"event_type": "action_rejected", "actor": 0, "component": "movement"},
        {
            "event_type": "ability_activated",
            "source": 1,
            "recipient": 6,
            "component": "ultimate",
        },
        {
            "event_type": "ability_activated",
            "source": 4,
            "recipient": 0,
            "component": "basic",
        },
        {
            "event_type": "source_damage_output",
            "source": 1,
            "recipient": 6,
            "raw": 15.0,
            "modified": 15.0,
            "modifier": 0.9,
            "mage_emitters": (),
            "warrior_emitters": (6,),
        },
        {
            "event_type": "source_healing_output",
            "source": 4,
            "recipient": 0,
            "raw": 10.0,
            "modified": 10.0,
            "modifier": 1.0,
        },
        {
            "event_type": "recipient_health_resolution",
            "recipient": 0,
            "before": 90.0,
            "after": 100.0,
            "delta": 10.0,
            "damage": 0.0,
            "healing": 10.0,
        },
        {
            "event_type": "recipient_health_resolution",
            "recipient": 6,
            "before": 10.0,
            "after": 0.0,
            "delta": -10.0,
            "damage": 13.5,
            "healing": 0.0,
        },
        {"event_type": "combat_countdown_reset", "agent": 1},
        {"event_type": "health_regenerated", "agent": 2, "amount": 2.0},
        {"event_type": "cooldown_started", "agent": 1},
        {"event_type": "cooldown_ready", "agent": 0},
        {"event_type": "charge_phase_displacement", "agent": 1},
        {"event_type": "ordinary_movement_phase_displacement", "agent": 1},
        {"event_type": "agent_died", "recipient": 6},
        {
            "event_type": "lethal_damage_contribution",
            "source": 1,
            "recipient": 6,
            "amount": 13.5,
        },
        {
            "event_type": "status_aged_to_zero",
            "recipient": 6,
            "status_id": "warrior_charge_stun",
        },
        {
            "event_type": "status_broken_by_damage",
            "recipient": 6,
            "status_id": "hunter_trap_stun",
        },
        {
            "event_type": "status_applied",
            "source": 4,
            "recipient": 0,
            "status_id": "priest_blessing_of_freedom_movement_floor",
        },
        {
            "event_type": "status_refreshed_or_extended",
            "recipient": 8,
            "status_id": "rogue_poison_slow",
        },
        {
            "event_type": "status_cleared_by_new_death",
            "recipient": 6,
            "status_id": "rogue_poison_anti_heal",
        },
        {"event_type": "spawn_shield_expired", "agent": 0},
        {"event_type": "respawn_wave_occurred", "team_index": 1},
        {"event_type": "agent_respawned", "agent": 9},
        {
            "event_type": "team_deathmatch_score_changed",
            "team_index": 0,
            "previous_score": 2,
            "score_increment": 1,
        },
        {
            "event_type": "team_deathmatch_completed",
            "outcome": "team_a_win",
            "completion_basis": "score_threshold_at_horizon",
        },
    )
)
_GRAMMAR_BATCH = _batch(
    _GRAMMAR_NAME,
    _GRAMMAR_AGENTS,
    _GRAMMAR_SPECS,
    start_positions={1: (5.0, 5.4)},
    post_charge_positions={1: (5.0, 5.8)},
)
_GRAMMAR_SCENE = _scene(
    _GRAMMAR_NAME,
    _GRAMMAR_AGENTS,
    event_batch=_GRAMMAR_BATCH,
    map_scene=MapSceneV1(width=16.0, height=12.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
    respawn_wave_countdowns=(4, 9),
)


_POV_CLASS_IDS = (
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
)
_POV_SOURCE_AGENTS = _agents(
    _MIXED_POSITIONS,
    status_tokens={
        0: ("stun_rogue_poison", "slow_rogue_poison", "anti_heal_rogue_poison"),
        5: ("stun_hunter_trap",),
    },
    health={0: 70.0, 5: 30.0},
    class_ids=_POV_CLASS_IDS,
    included_slots=(0, 1, 5),
)
_POV_SOURCE_SPECS = (
    *_activation_specs((("basic_damage", 0, 5), ("rogue_poison", 5, 0))),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 0,
            "before": 80.0,
            "after": 70.0,
            "delta": -10.0,
            "damage": 10.0,
            "healing": 0.0,
        }
    ),
    MappingProxyType(
        {
            "event_type": "recipient_health_resolution",
            "recipient": 5,
            "before": 37.0,
            "after": 30.0,
            "delta": -7.0,
            "damage": 7.0,
            "healing": 0.0,
        }
    ),
    *(
        MappingProxyType(
            {
                "event_type": "status_applied",
                "source": 5,
                "recipient": 0,
                "status_id": status_id,
            }
        )
        for status_id in (
            "rogue_poison_stun",
            "rogue_poison_slow",
            "rogue_poison_anti_heal",
        )
    ),
)
_POV_SOURCE_BATCH = _batch(
    "pov_redaction_source", _POV_SOURCE_AGENTS, _POV_SOURCE_SPECS
)
_POV_SOURCE_SCENE = _scene(
    "pov_redaction_source",
    _POV_SOURCE_AGENTS,
    event_batch=_POV_SOURCE_BATCH,
    map_scene=MapSceneV1(width=10.0, height=12.0),
    selection=SelectionSceneV1(controlled_global_slot=0, selected_global_slot=5),
    visibility_by_slot={0: True, 1: True, 5: False},
)

_POV_EPISODE = "synthetic:pov_redaction"
_POV_PUBLIC_ID = "0"
_POV_TRANSITION = f"{_POV_EPISODE}:actor-pov:{_POV_PUBLIC_ID}:transition:0"
_POV_SCENE = ActorPovBattlefieldSceneV1(
    schema_version=ACTOR_POV_SCENE_SCHEMA_VERSION,
    audience_badge="AGENT POV · EXACT · SYNTHETIC FIXTURE",
    observation_materialization="exact_no_shared_obs_actor_input",
    episode_id=_POV_EPISODE,
    frame_index=1,
    pov_frame_id=f"{_POV_EPISODE}:actor-pov:{_POV_PUBLIC_ID}:frame:1",
    source_frame_id=f"{_POV_EPISODE}:frame:1",
    simulator_step_count=1,
    map=MapSceneV1(width=10.0, height=12.0),
    self_actor=ActorPovSelfSceneV1(
        global_slot=0,
        public_agent_id=_POV_PUBLIC_ID,
        team_local_slot=0,
        team_id=1,
        class_id=MAGE_CLASS_ID,
        position=_MIXED_POSITIONS[0],
        radius=0.5,
        alive=True,
        current_health=70.0,
        max_health=100.0,
        effective_movement_speed=1.0,
        ultimate_cooldown_remaining=2,
        steps_until_out_of_combat=1,
        spawn_shield_remaining=0,
        status_feature_values=(
            1.0,
            1.0,
            1.0,
            0.5,
            0.85,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    visible_bodies=(
        ActorPovVisibleBodySceneV1(
            relation="ally",
            observation_row=1,
            public_agent_id="1",
            position=_MIXED_POSITIONS[1],
            radius=0.5,
            team_id=1,
            class_id=PRIEST_CLASS_ID,
            alive=True,
            current_health=100.0,
            max_health=100.0,
            effective_movement_speed=1.0,
            ultimate_cooldown_remaining=0,
            steps_until_out_of_combat=0,
            status_feature_values=(
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                2.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        ),
    ),
    spawn_pads=(
        ActorPovSpawnPadSceneV1(
            actor_relative_team_index=0,
            team_relation="own",
            team_label="Own Team",
            team_local_slot=0,
            position=_MIXED_POSITIONS[0],
            configured_active=True,
            currently_alive=True,
            spawn_shield_remaining=0,
        ),
        ActorPovSpawnPadSceneV1(
            actor_relative_team_index=0,
            team_relation="own",
            team_label="Own Team",
            team_local_slot=1,
            position=_MIXED_POSITIONS[1],
            configured_active=True,
            currently_alive=True,
            spawn_shield_remaining=0,
        ),
    ),
    respawn_waves=(
        ActorPovRespawnWaveSceneV1(
            actor_relative_team_index=0,
            team_relation="own",
            team_label="Own Team",
            period_steps=10,
            countdown_steps=4,
        ),
        ActorPovRespawnWaveSceneV1(
            actor_relative_team_index=1,
            team_relation="opponent",
            team_label="Opponent Team",
            period_steps=10,
            countdown_steps=7,
        ),
    ),
)
_POV_JOINT_MASK = ((True, True),) + ((False, False),) * 10
_POV_ACTION_MASK = ActorPovActionMaskV1(
    move=(True,) * 9,
    select_target=(True,) + (False,) * 10,
    use_ultimate=(True, True),
    select_target_use_ultimate_joint=_POV_JOINT_MASK,
)
_POV_TARGET_PUBLIC_AGENT_IDS: tuple[str | None, ...] = (
    None,
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
_POV_CUES = (
    ActorPovOwnActionOutcomeCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:0",
        pov_transition_id=_POV_TRANSITION,
        ordinal=0,
        outcome="accepted",
    ),
    ActorPovOwnPositionChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:1",
        pov_transition_id=_POV_TRANSITION,
        ordinal=1,
        start_position=(2.5, 4.0),
        successor_position=_MIXED_POSITIONS[0],
    ),
    ActorPovOwnHealthChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:2",
        pov_transition_id=_POV_TRANSITION,
        ordinal=2,
        start_health=80.0,
        successor_health=70.0,
    ),
    ActorPovOwnStatusChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:3",
        pov_transition_id=_POV_TRANSITION,
        ordinal=3,
        changed_feature_indices=(15,),
        start_values=(0.0,),
        successor_values=(1.0,),
    ),
    ActorPovOwnStatusChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:4",
        pov_transition_id=_POV_TRANSITION,
        ordinal=4,
        changed_feature_indices=(16,),
        start_values=(0.0,),
        successor_values=(1.0,),
    ),
    ActorPovOwnStatusChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:5",
        pov_transition_id=_POV_TRANSITION,
        ordinal=5,
        changed_feature_indices=(17,),
        start_values=(0.0,),
        successor_values=(1.0,),
    ),
)
_POV_PROJECTION = ActorPovAnalyzerProjectionV1(
    scene=_POV_SCENE,
    next_decision_action_mask=_POV_ACTION_MASK,
    incoming_transition_id=_POV_TRANSITION,
    incoming_cues=_POV_CUES,
)

_POV_EXHAUSTIVE_CUES = (
    ActorPovOwnActionOutcomeCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:0",
        pov_transition_id=_POV_TRANSITION,
        ordinal=0,
        outcome="accepted",
    ),
    ActorPovOwnPositionChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:1",
        pov_transition_id=_POV_TRANSITION,
        ordinal=1,
        start_position=(2.5, 4.0),
        successor_position=_MIXED_POSITIONS[0],
    ),
    ActorPovOwnHealthChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:2",
        pov_transition_id=_POV_TRANSITION,
        ordinal=2,
        start_health=80.0,
        successor_health=70.0,
    ),
    ActorPovOwnStatusChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:3",
        pov_transition_id=_POV_TRANSITION,
        ordinal=3,
        changed_feature_indices=(15, 16, 17),
        start_values=(0.0, 0.0, 0.0),
        successor_values=(1.0, 1.0, 1.0),
    ),
    ActorPovOwnCooldownChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:4",
        pov_transition_id=_POV_TRANSITION,
        ordinal=4,
        start_remaining_ticks=3.0,
        successor_remaining_ticks=2.0,
    ),
    ActorPovOwnLifecycleChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:5",
        pov_transition_id=_POV_TRANSITION,
        ordinal=5,
        start_active=True,
        successor_active=True,
        start_alive=True,
        successor_alive=True,
        start_spawn_shield_remaining_ticks=2,
        successor_spawn_shield_remaining_ticks=0,
    ),
    ActorPovVisibleBodyObservationChangedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:6",
        pov_transition_id=_POV_TRANSITION,
        ordinal=6,
        relation="ally",
        observation_row=1,
        start_visible=False,
        successor_visible=True,
        observed_payload_changed=True,
    ),
    ActorPovEpisodeEndedCueV1(
        cue_id=f"{_POV_TRANSITION}:cue:7",
        pov_transition_id=_POV_TRANSITION,
        ordinal=7,
        terminated=True,
        truncated=False,
        public_end_reason="synthetic fixture complete",
    ),
)
_POV_EXHAUSTIVE_PROJECTION = ActorPovAnalyzerProjectionV1(
    scene=_POV_SCENE,
    next_decision_action_mask=_POV_ACTION_MASK,
    incoming_transition_id=_POV_TRANSITION,
    incoming_cues=_POV_EXHAUSTIVE_CUES,
)


def _terminal_state() -> TerminalStateV2:
    return TerminalStateV2(
        is_sealed=False,
        terminated=False,
        truncated=False,
        reached_declared_horizon=False,
        reason=None,
    )


def _scenario_metadata(
    name: RendererFixtureName,
    description: str,
    *,
    frame_index: int,
) -> ScenarioMetadataV1:
    return ScenarioMetadataV1(
        name=name,
        title=f"SYNTHETIC · {name}",
        description=description,
        mode="scripted",
        audience="researcher",
        movement_scale_minimum=0.01,
        movement_scale_maximum=1.0,
        movement_scale_step=0.01,
        ordinary_movement_distance_scale=1.0,
        scenario_default_movement_scale=1.0,
        movement_scale_overridden=False,
        completed_frame_count=frame_index,
        frame_count=frame_index,
        next_frame_index=None,
        next_frame_label=None,
        next_frame_description=None,
        script_complete=True,
    )


def _scenario_option(
    name: RendererFixtureName,
    description: str,
) -> ScenarioOptionV1:
    return ScenarioOptionV1(
        name=name,
        title=f"SYNTHETIC · {name}",
        description=description,
        mode="scripted",
        audience="researcher",
    )


def _status_source_state(scene: BattlefieldSceneV2) -> StatusSourceEvidenceStateV2:
    rows = tuple(
        StatusSourceChannelEvidenceV2(
            recipient_global_slot=agent.global_slot,
            recipient_public_agent_id=agent.public_agent_id,
            status_channel=status.status_channel,
            status_id=status.status_id,
            direct_source_evidence=status.direct_source_evidence,
        )
        for agent in scene.agents
        for status in agent.statuses
    )
    return StatusSourceEvidenceStateV2(
        schema_version=2,
        episode_id=scene.episode_id,
        frame_index=scene.frame_index,
        frame_id=scene.frame_id,
        active_statuses=tuple(
            sorted(
                rows,
                key=lambda row: (row.recipient_global_slot, row.status_channel),
            )
        ),
    )


def _researcher_live_frame(
    name: RendererFixtureName,
    description: str,
    scene: BattlefieldSceneV2,
    event_batch: VisualEventBatchV2 | None,
) -> ResearcherLiveDebuggerFrameV2:
    selection = scene.selection
    if selection is None:
        raise ValueError("synthetic researcher scenes require an exact selection.")
    pending = PendingActionCardV1(
        label="PLAYBACK / INSPECTION ONLY",
        actor_global_slot=selection.controlled_global_slot,
        move_action=0,
        target_action=0,
        armed_lane=None,
        arm_origin=None,
        target=TargetReferenceV1(disclosure="target_none", global_slot=None),
        movement_mask_value=True,
        pair_mask_value=None,
        summary="STAY + NO COMBAT",
    )
    scenario = _scenario_metadata(name, description, frame_index=scene.frame_index)
    return ResearcherLiveDebuggerFrameV2(
        session_id=f"synthetic_{name}",
        run_generation=0,
        revision=0,
        episode_id=scene.episode_id,
        frame_index=scene.frame_index,
        frame_id=scene.frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_transition_index=(
            None if scene.frame_index == 0 else scene.frame_index - 1
        ),
        incoming_transition_id=scene.incoming_transition_id,
        preset="analysis",
        verbose=False,
        show_ranges=bool(scene.ranges),
        terminal=_terminal_state(),
        scenario=scenario,
        available_scenarios=(_scenario_option(name, description),),
        projection=ResearcherAnalyzerProjectionV2(
            schema_version=2,
            scene=scene,
            incoming_events=event_batch,
            status_source_evidence=_status_source_state(scene),
        ),
        hud=ResearcherHudFrameV2(
            roster_global_slots=tuple(agent.global_slot for agent in scene.agents),
            controlled_global_slot=selection.controlled_global_slot,
            selected_global_slot=selection.selected_global_slot,
            pending_submission_scope="scripted_playback",
            pending_actions=(pending,),
            pending_action=pending,
            latest_transition=None,
            movement_legalities=tuple(
                MovementLegalityCardV1(move_action=move_action, available=True)
                for move_action in range(9)
            ),
            candidate_legalities=(),
            diagnostics=(),
        ),
    )


def fixture_pov_target_reference_v1(
    target_action: int,
    target_public_agent_ids: tuple[str | None, ...],
) -> ActorPovTargetReferenceV1:
    """Resolve a synthetic POV target through its explicit recipient-local axis."""
    if type(target_action) is not int or not 0 <= target_action < len(
        target_public_agent_ids
    ):
        raise ValueError("target_action is outside the supplied POV target axis.")
    return ActorPovTargetReferenceV1(
        target_action=target_action,
        public_agent_id=target_public_agent_ids[target_action],
    )


def _pov_live_frame(
    name: RendererFixtureName,
    projection: ActorPovAnalyzerProjectionV1,
    target_public_agent_ids: tuple[str | None, ...],
    *,
    terminal: TerminalStateV2 | None = None,
) -> ActorPovLiveDebuggerFrameV2:
    scene = projection.scene
    mask = projection.next_decision_action_mask
    pending_target = fixture_pov_target_reference_v1(0, target_public_agent_ids)
    return ActorPovLiveDebuggerFrameV2(
        session_id=f"synthetic_{name}",
        run_generation=0,
        revision=0,
        episode_id=scene.episode_id,
        frame_index=scene.frame_index,
        frame_id=scene.source_frame_id,
        simulator_step_count=scene.simulator_step_count,
        preset="analysis",
        verbose=False,
        terminal=_terminal_state() if terminal is None else terminal,
        incoming_pov_transition_id=projection.incoming_transition_id,
        projection=projection,
        hud=ActorPovHudFrameV1(
            controlled_public_agent_id=scene.self_actor.public_agent_id,
            pending_submission_scope="scripted_playback",
            pending_action=ActorPovPendingActionCardV1(
                label="PLAYBACK / INSPECTION ONLY",
                actor_public_agent_id=scene.self_actor.public_agent_id,
                move_action=0,
                target=pending_target,
                armed_lane=None,
                arm_origin=None,
                movement_mask_value=mask.move[0],
                pair_mask_value=None,
                summary="STAY + NO COMBAT",
            ),
            latest_transition=None,
            movement_legalities=tuple(
                MovementLegalityCardV1(
                    move_action=move_action,
                    available=available,
                )
                for move_action, available in enumerate(mask.move)
            ),
            candidate_legalities=tuple(
                ActorPovCandidateLegalityCardV1(
                    target=fixture_pov_target_reference_v1(
                        target_action,
                        target_public_agent_ids,
                    ),
                    lane_0_available=lanes[0],
                    lane_1_available=lanes[1],
                    basic_available=target_action > 0 and lanes[0],
                    ultimate_available=lanes[1],
                )
                for target_action, lanes in enumerate(
                    mask.select_target_use_ultimate_joint
                )
            ),
            diagnostics=(),
        ),
    )


def _researcher_fixture(
    *,
    name: RendererFixtureName,
    description: str,
    scene: BattlefieldSceneV2,
    event_batch: VisualEventBatchV2 | None,
    viewports: tuple[ViewportCaseV1, ...] = (),
    exercise_reduced_motion: bool = False,
    with_presentation_pair: bool = False,
) -> RendererFixtureV2:
    live_frame = _researcher_live_frame(name, description, scene, event_batch)
    return RendererFixtureV2(
        name=name,
        description=description,
        audience="researcher",
        scene=scene,
        live_frame=live_frame,
        event_batch=event_batch,
        viewports=viewports,
        exercise_reduced_motion=exercise_reduced_motion,
        synthetic_presentation_pair=(
            _synthetic_researcher_presentation_pair(live_frame)
            if with_presentation_pair
            else None
        ),
    )


_POV_EXHAUSTIVE_LIVE_FRAME = _pov_live_frame(
    "pov_redaction",
    _POV_EXHAUSTIVE_PROJECTION,
    _POV_TARGET_PUBLIC_AGENT_IDS,
    terminal=TerminalStateV2(
        is_sealed=True,
        terminated=True,
        truncated=False,
        reached_declared_horizon=True,
        reason="terminated",
    ),
)
_POV_PRESENTATION_PAIR = _synthetic_pov_presentation_pair(_POV_EXHAUSTIVE_LIVE_FRAME)


RENDERER_FIXTURES: Mapping[str, RendererFixtureV2] = MappingProxyType(
    {
        "visual_vocabulary": _researcher_fixture(
            name="visual_vocabulary",
            description=(
                "SYNTHETIC: class identities, ranges, cooldowns, and canonical "
                "Basic/Ultimate activation grammar."
            ),
            scene=_VOCABULARY_SCENE,
            event_batch=_VOCABULARY_BATCH,
            with_presentation_pair=True,
        ),
        "durable_controls": _researcher_fixture(
            name="durable_controls",
            description=(
                "SYNTHETIC: canonical stun and slow duration glyphs with "
                "source-class accents."
            ),
            scene=_DURABLE_SCENE,
            event_batch=None,
        ),
        "crowded_teamfight": _researcher_fixture(
            name="crowded_teamfight",
            description=(
                "SYNTHETIC: dense V2 status, aura, range, selection, legality, "
                "and simultaneous-event pressure."
            ),
            scene=_CROWDED_SCENE,
            event_batch=_CROWDED_BATCH,
            with_presentation_pair=True,
        ),
        "required_dock_fallback": _researcher_fixture(
            name="required_dock_fallback",
            description=(
                "SYNTHETIC: near-dense agents demonstrate individually owned "
                "compact cooldown fallbacks."
            ),
            scene=_REQUIRED_DOCK_SCENE,
            event_batch=None,
        ),
        "route_collision": _researcher_fixture(
            name="route_collision",
            description=(
                "SYNTHETIC: reciprocal, parallel, crossing, and near-zero "
                "canonical activation routes."
            ),
            scene=_ROUTE_SCENE,
            event_batch=_ROUTE_BATCH,
        ),
        "mixed_net_zero": _researcher_fixture(
            name="mixed_net_zero",
            description=(
                "SYNTHETIC: damage and healing activations with one exact "
                "zero-net recipient resolution."
            ),
            scene=_MIXED_SCENE,
            event_batch=_MIXED_BATCH,
            with_presentation_pair=True,
        ),
        "viewport_matrix": _researcher_fixture(
            name="viewport_matrix",
            description=(
                "SYNTHETIC: crowded V2 scene at all supported responsive and "
                "reduced-motion settings."
            ),
            scene=_VIEWPORT_SCENE,
            event_batch=_VIEWPORT_BATCH,
            viewports=_VIEWPORTS,
            exercise_reduced_motion=True,
        ),
        "canonical_event_vocabulary": _researcher_fixture(
            name="canonical_event_vocabulary",
            description=(
                "SYNTHETIC: one gap-free ordered instance of every canonical "
                "V2 event variant."
            ),
            scene=_GRAMMAR_SCENE,
            event_batch=_GRAMMAR_BATCH,
            with_presentation_pair=True,
        ),
        "pov_redaction": RendererFixtureV2(
            name="pov_redaction",
            description=(
                "SYNTHETIC: independent recipient-local POV scene/cues beside "
                "a private V2 comparison source."
            ),
            audience="agent_pov",
            scene=_POV_SCENE,
            live_frame=_pov_live_frame(
                "pov_redaction",
                _POV_PROJECTION,
                _POV_TARGET_PUBLIC_AGENT_IDS,
            ),
            pov_projection=_POV_PROJECTION,
            pov_target_public_agent_ids=_POV_TARGET_PUBLIC_AGENT_IDS,
            privileged_source_scene=_POV_SOURCE_SCENE,
            privileged_source_event_batch=_POV_SOURCE_BATCH,
            synthetic_presentation_pair=_POV_PRESENTATION_PAIR,
        ),
    }
)


def get_renderer_fixture(name: str) -> RendererFixtureV2:
    """Return a synthetic renderer fixture by stable name."""
    try:
        return RENDERER_FIXTURES[name]
    except KeyError as exc:
        choices = ", ".join(RENDERER_FIXTURES)
        raise ValueError(
            f"unknown renderer fixture {name!r}; choose one of: {choices}."
        ) from exc


def list_renderer_fixtures() -> tuple[RendererFixtureV2, ...]:
    """Return synthetic renderer fixtures in deterministic review order."""
    return tuple(RENDERER_FIXTURES.values())
