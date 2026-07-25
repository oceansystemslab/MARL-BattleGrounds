"""SYNTHETIC renderer-only fixtures for deterministic browser layout tests.

These records are deliberately not simulator scenarios. They never construct
or submit actions, and they must not be presented as valid simulator history.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
)
from marl_battlegrounds.rendering.scene import (
    EVENT_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    AcceptedActivationEventV1,
    AgentSceneV1,
    AuraFieldSceneV1,
    BattlefieldSceneV1,
    ChargeDisplacementEventV1,
    HealthOutcome,
    MapSceneV1,
    ModifierSceneV1,
    NetHealthEventV1,
    ObserverVisibilitySceneV1,
    ObstacleSceneV1,
    PendingRouteSceneV1,
    RangeSceneV1,
    SelectedLegalitySceneV1,
    SelectionSceneV1,
    StatusLifecycleEventV1,
    StatusSceneV1,
    VisualEventBatchV1,
)
from marl_battlegrounds.rendering.vocabulary import (
    CANONICAL_STATUS_ORDER,
    ActivationTokenId,
    StatusTokenId,
    lookup_status_token,
)

type RendererFixtureName = Literal[
    "crowded_teamfight",
    "route_collision",
    "mixed_net_zero",
    "viewport_matrix",
    "pov_redaction",
]
type ViewportLabel = Literal[
    "desktop",
    "compact",
    "minimum",
    "stacked",
]
type ViewportLayout = Literal["split", "stacked"]
type SyntheticEventV1 = (
    AcceptedActivationEventV1
    | NetHealthEventV1
    | ChargeDisplacementEventV1
    | StatusLifecycleEventV1
)


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
class RendererFixtureV1:
    """One explicitly synthetic, already-authorized presentation payload."""

    name: RendererFixtureName
    description: str
    scene: BattlefieldSceneV1
    event_batch: VisualEventBatchV1 | None = None
    viewports: tuple[ViewportCaseV1, ...] = ()
    exercise_reduced_motion: bool = False
    privileged_source_scene: BattlefieldSceneV1 | None = None
    privileged_source_event_batch: VisualEventBatchV1 | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("fixture name must be a non-empty Python string.")
        if type(self.description) is not str or not self.description.startswith(
            "SYNTHETIC:"
        ):
            raise ValueError("fixture description must begin with 'SYNTHETIC:'.")
        if type(self.scene) is not BattlefieldSceneV1:
            raise ValueError("fixture scene must be BattlefieldSceneV1.")
        if (
            self.event_batch is not None
            and type(self.event_batch) is not VisualEventBatchV1
        ):
            raise ValueError("event_batch must be VisualEventBatchV1 or None.")
        if type(self.viewports) is not tuple or any(
            type(viewport) is not ViewportCaseV1 for viewport in self.viewports
        ):
            raise ValueError("viewports must be a tuple of ViewportCaseV1 records.")
        labels = tuple(viewport.label for viewport in self.viewports)
        if len(labels) != len(set(labels)):
            raise ValueError("viewport labels must be unique.")
        if type(self.exercise_reduced_motion) is not bool:
            raise ValueError("exercise_reduced_motion must be a Python bool.")
        if (
            self.privileged_source_scene is not None
            and type(self.privileged_source_scene) is not BattlefieldSceneV1
        ):
            raise ValueError(
                "privileged_source_scene must be BattlefieldSceneV1 or None."
            )
        if (
            self.privileged_source_event_batch is not None
            and type(self.privileged_source_event_batch) is not VisualEventBatchV1
        ):
            raise ValueError(
                "privileged_source_event_batch must be VisualEventBatchV1 or None."
            )
        if (self.privileged_source_scene is None) is not (
            self.privileged_source_event_batch is None
        ):
            raise ValueError(
                "privileged source scene and event batch must be supplied together."
            )
        if self.privileged_source_scene is not None:
            if self.privileged_source_scene.audience != "researcher":
                raise ValueError("privileged source scenes must be researcher scenes.")
            if self.scene.audience != "agent_pov":
                raise ValueError(
                    "fixtures with a privileged source must render an agent-POV scene."
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


def _status(token_id: StatusTokenId, duration: int) -> StatusSceneV1:
    definition = lookup_status_token(token_id)
    if definition.source_class_id is None:
        raise AssertionError(f"status {token_id!r} must have a source class")
    return StatusSceneV1(
        token_id=token_id,
        duration=duration,
        source_class_id=definition.source_class_id,
        label=definition.label,
        short_label=definition.short_label,
        accessible_name=definition.accessible_name,
        priority=definition.priority,
    )


def _statuses(token_ids: tuple[StatusTokenId, ...]) -> tuple[StatusSceneV1, ...]:
    ordered: tuple[StatusTokenId, ...] = tuple(
        sorted(token_ids, key=CANONICAL_STATUS_ORDER.index)
    )
    duration_by_token = dict(
        zip(
            CANONICAL_STATUS_ORDER,
            (3, 3, 3, 2, 2, 2, 3, 2, 3),
            strict=True,
        )
    )
    return tuple(
        _status(token_id, duration=duration_by_token[token_id]) for token_id in ordered
    )


def _modifier(
    token_id: str,
    multiplier: float,
    label: str,
    accessible_name: str,
) -> ModifierSceneV1:
    return ModifierSceneV1(
        token_id=token_id,
        multiplier=multiplier,
        label=label,
        accessible_name=accessible_name,
    )


def _agents(
    positions: tuple[tuple[float, float], ...],
    *,
    status_tokens: Mapping[int, tuple[StatusTokenId, ...]] | None = None,
    health: Mapping[int, float] | None = None,
    modifiers: Mapping[int, tuple[ModifierSceneV1, ...]] | None = None,
    class_ids: tuple[int, ...] = _CLASS_IDS,
    included_slots: tuple[int, ...] = tuple(range(10)),
) -> tuple[AgentSceneV1, ...]:
    if len(positions) != 10:
        raise ValueError("synthetic position tables require ten fixed-slot rows.")
    if len(class_ids) != 10:
        raise ValueError("synthetic class tables require ten fixed-slot rows.")
    status_tokens = {} if status_tokens is None else status_tokens
    health = {} if health is None else health
    modifiers = {} if modifiers is None else modifiers
    return tuple(
        AgentSceneV1(
            global_slot=slot,
            team_id=_TEAM_IDS[slot],
            class_id=class_ids[slot],
            position=positions[slot],
            radius=0.5,
            active=True,
            alive=True,
            current_health=health.get(slot, 100.0),
            max_health=100.0,
            ultimate_cooldown=0,
            effective_speed=1.0,
            statuses=_statuses(status_tokens.get(slot, ())),
            modifiers=modifiers.get(slot, ()),
        )
        for slot in included_slots
    )


def _activation(
    *,
    fixture_name: str,
    ordinal: int,
    token_id: ActivationTokenId,
    source_global_slot: int,
    target_global_slot: int | None,
    agents: Mapping[int, AgentSceneV1],
    anchor_positions: Mapping[int, tuple[float, float]] | None = None,
) -> AcceptedActivationEventV1:
    source = agents[source_global_slot]
    target = None if target_global_slot is None else agents[target_global_slot]
    anchor_positions = {} if anchor_positions is None else anchor_positions
    if target is None:
        target_anchor = None
    else:
        if target_global_slot is None:
            raise AssertionError("target object requires a target global slot")
        target_anchor = anchor_positions.get(target_global_slot, target.position)
    return AcceptedActivationEventV1(
        event_id=f"synthetic:{fixture_name}:activation-{ordinal}",
        transition_id=1,
        token_id=token_id,
        source_global_slot=source_global_slot,
        target_global_slot=target_global_slot,
        source_anchor=anchor_positions.get(source_global_slot, source.position),
        target_anchor=target_anchor,
        target_disclosure=("target_none" if target_global_slot is None else "public"),
        lane=0 if token_id in ("basic_damage", "basic_heal") else 1,
        source_class_id=source.class_id,
    )


def _batch(
    events: tuple[SyntheticEventV1, ...],
) -> VisualEventBatchV1:
    return VisualEventBatchV1(
        schema_version=EVENT_SCHEMA_VERSION,
        transition_id=1,
        simulator_step=1,
        events=events,
    )


_CROWDED_POSITIONS = (
    (4.6, 4.4),
    (5.3, 5.0),
    (6.8, 4.1),
    (8.1, 4.3),
    (9.4, 4.1),
    (4.8, 6.2),
    (6.5, 5.0),
    (6.9, 6.2),
    (8.2, 6.0),
    (9.5, 6.2),
)
_CROWDED_PRE_ANCHORS: Mapping[int, tuple[float, float]] = MappingProxyType(
    {
        1: (5.3, 4.1),
        6: (6.5, 5.9),
    }
)
_CROWDED_STATUS_TOKENS: Mapping[int, tuple[StatusTokenId, ...]] = MappingProxyType(
    {slot: CANONICAL_STATUS_ORDER for slot in range(10)}
)
_CROWDED_AGENT_MODIFIERS = (
    _modifier(
        "mage_amplification",
        1.2,
        "Mage aura x1.20",
        "Effective Mage amplification multiplier 1.20",
    ),
    _modifier(
        "warrior_mitigation",
        0.8,
        "Warrior aura x0.80",
        "Effective Warrior mitigation multiplier 0.80",
    ),
)
_CROWDED_MODIFIERS: Mapping[
    int,
    tuple[ModifierSceneV1, ...],
] = MappingProxyType({slot: _CROWDED_AGENT_MODIFIERS for slot in range(10)})
_CROWDED_HEALTH: Mapping[int, float] = MappingProxyType(
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
    modifiers=_CROWDED_MODIFIERS,
)
_CROWDED_AGENT_MAP = {agent.global_slot: agent for agent in _CROWDED_AGENTS}
_CROWDED_SCENE = BattlefieldSceneV1(
    schema_version=SCENE_SCHEMA_VERSION,
    audience="researcher",
    audience_badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC FIXTURE",
    map=MapSceneV1(
        width=16.0,
        height=12.0,
        obstacles=(
            ObstacleSceneV1(
                obstacle_id="synthetic-crowded-pillar",
                kind="pillar",
                center=(2.0, 9.5),
                radius=0.7,
            ),
            ObstacleSceneV1(
                obstacle_id="synthetic-crowded-wall",
                kind="wall",
                center=(13.0, 6.0),
                width=0.8,
                height=4.0,
                theta=0.0,
            ),
        ),
    ),
    agents=_CROWDED_AGENTS,
    aura_fields=(
        AuraFieldSceneV1(
            source_global_slot=0,
            token_id="mage_amplification",
            center=_CROWDED_POSITIONS[0],
            radius=4.0,
        ),
        AuraFieldSceneV1(
            source_global_slot=1,
            token_id="warrior_mitigation",
            center=_CROWDED_POSITIONS[1],
            radius=4.0,
        ),
        AuraFieldSceneV1(
            source_global_slot=5,
            token_id="mage_amplification",
            center=_CROWDED_POSITIONS[5],
            radius=4.0,
        ),
        AuraFieldSceneV1(
            source_global_slot=6,
            token_id="warrior_mitigation",
            center=_CROWDED_POSITIONS[6],
            radius=4.0,
        ),
    ),
    ranges=(
        RangeSceneV1(
            global_slot=0,
            center=_CROWDED_POSITIONS[0],
            radius=6.0,
            kind="observation",
        ),
        RangeSceneV1(
            global_slot=0,
            center=_CROWDED_POSITIONS[0],
            radius=3.0,
            kind="basic",
        ),
        RangeSceneV1(
            global_slot=0,
            center=_CROWDED_POSITIONS[0],
            radius=4.0,
            kind="ultimate",
        ),
    ),
    selection=SelectionSceneV1(
        controlled_global_slot=0,
        selected_global_slot=7,
    ),
    selected_legality=SelectedLegalitySceneV1(
        controlled_global_slot=0,
        target_global_slot=7,
        target_action=8,
        lane_0_available=True,
        lane_1_available=False,
        armed_lane=1,
        armed_pair_legal=False,
    ),
    pending_route=PendingRouteSceneV1(
        source_global_slot=0,
        target_global_slot=7,
        source_anchor=_CROWDED_POSITIONS[0],
        target_anchor=_CROWDED_POSITIONS[7],
        lane=1,
        legal=False,
    ),
    observer_visibility=tuple(
        ObserverVisibilitySceneV1(
            observer_global_slot=0,
            candidate_global_slot=slot,
            visible=slot < 7,
        )
        for slot in range(10)
    ),
)
_CROWDED_ACTIVATION_SPECS: tuple[
    tuple[ActivationTokenId, int, int | None],
    ...,
] = (
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
_CROWDED_ACTIVATIONS = tuple(
    _activation(
        fixture_name="crowded_teamfight",
        ordinal=ordinal,
        token_id=token_id,
        source_global_slot=source,
        target_global_slot=target,
        agents=_CROWDED_AGENT_MAP,
        anchor_positions=_CROWDED_PRE_ANCHORS,
    )
    for ordinal, (token_id, source, target) in enumerate(_CROWDED_ACTIVATION_SPECS)
)
_CROWDED_NET_SPECS: tuple[
    tuple[int, float, float, HealthOutcome],
    ...,
] = (
    (0, 90.0, 82.0, "damage"),
    (1, 100.0, 88.0, "damage"),
    (3, 100.0, 92.0, "damage"),
    (4, 70.0, 80.0, "healing"),
    (5, 90.0, 82.0, "damage"),
    (6, 100.0, 88.0, "damage"),
    (8, 100.0, 92.0, "damage"),
    (9, 70.0, 80.0, "healing"),
)
_CROWDED_NET_EVENTS = tuple(
    NetHealthEventV1(
        event_id=f"synthetic:crowded_teamfight:net-health-{ordinal}",
        transition_id=1,
        recipient_global_slot=slot,
        recipient_anchor=_CROWDED_POSITIONS[slot],
        health_before=health_before,
        health_after=health_after,
        net_delta=health_after - health_before,
        outcome=outcome,
    )
    for ordinal, (slot, health_before, health_after, outcome) in enumerate(
        _CROWDED_NET_SPECS
    )
)
_CROWDED_CHARGE_EVENTS = (
    ChargeDisplacementEventV1(
        event_id="synthetic:crowded_teamfight:charge-0",
        transition_id=1,
        source_global_slot=1,
        target_global_slot=6,
        start=_CROWDED_PRE_ANCHORS[1],
        end=_CROWDED_POSITIONS[1],
        path_kind="charge_only",
    ),
    ChargeDisplacementEventV1(
        event_id="synthetic:crowded_teamfight:charge-1",
        transition_id=1,
        source_global_slot=6,
        target_global_slot=1,
        start=_CROWDED_PRE_ANCHORS[6],
        end=_CROWDED_POSITIONS[6],
        path_kind="charge_only",
    ),
)
_CROWDED_LIFECYCLE_SPECS: tuple[
    tuple[int, StatusTokenId, int, int],
    ...,
] = (
    (6, "stun_warrior_charge", 2, WARRIOR_CLASS_ID),
    (6, "slow_warrior_charge", 2, WARRIOR_CLASS_ID),
    (1, "stun_warrior_charge", 3, WARRIOR_CLASS_ID),
    (1, "slow_warrior_charge", 3, WARRIOR_CLASS_ID),
    (7, "stun_hunter_trap", 4, HUNTER_CLASS_ID),
    (2, "stun_hunter_trap", 5, HUNTER_CLASS_ID),
    (8, "stun_rogue_poison", 6, ROGUE_CLASS_ID),
    (8, "slow_rogue_poison", 6, ROGUE_CLASS_ID),
    (8, "anti_heal_rogue_poison", 6, ROGUE_CLASS_ID),
    (3, "stun_rogue_poison", 7, ROGUE_CLASS_ID),
    (3, "slow_rogue_poison", 7, ROGUE_CLASS_ID),
    (3, "anti_heal_rogue_poison", 7, ROGUE_CLASS_ID),
)
_CROWDED_LIFECYCLE_EVENTS = tuple(
    StatusLifecycleEventV1(
        event_id=f"synthetic:crowded_teamfight:status-{ordinal}",
        transition_id=1,
        recipient_global_slot=recipient,
        recipient_anchor=_CROWDED_POSITIONS[recipient],
        token_id=token_id,
        change="applied",
        duration_before=0,
        duration_after=next(
            status.duration
            for status in _CROWDED_AGENT_MAP[recipient].statuses
            if status.token_id == token_id
        ),
        source_class_id=source_class_id,
        application_event_ids=(_CROWDED_ACTIVATIONS[activation_index].event_id,),
    )
    for ordinal, (
        recipient,
        token_id,
        activation_index,
        source_class_id,
    ) in enumerate(_CROWDED_LIFECYCLE_SPECS)
)
_CROWDED_BATCH = _batch(
    (
        *_CROWDED_ACTIVATIONS,
        *_CROWDED_NET_EVENTS,
        *_CROWDED_CHARGE_EVENTS,
        *_CROWDED_LIFECYCLE_EVENTS,
    )
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
_ROUTE_AGENTS = _agents(
    _ROUTE_POSITIONS,
    class_ids=(HUNTER_CLASS_ID,) * 10,
)
_ROUTE_AGENT_MAP = {agent.global_slot: agent for agent in _ROUTE_AGENTS}
_ROUTE_SCENE = BattlefieldSceneV1(
    schema_version=SCENE_SCHEMA_VERSION,
    audience="researcher",
    audience_badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC FIXTURE",
    map=MapSceneV1(width=12.0, height=10.0),
    agents=_ROUTE_AGENTS,
    selection=SelectionSceneV1(
        controlled_global_slot=0,
        selected_global_slot=5,
    ),
)
_ROUTE_SPECS: tuple[tuple[ActivationTokenId, int, int | None], ...] = (
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
_ROUTE_BATCH = _batch(
    tuple(
        _activation(
            fixture_name="route_collision",
            ordinal=ordinal,
            token_id=token_id,
            source_global_slot=source,
            target_global_slot=target,
            agents=_ROUTE_AGENT_MAP,
        )
        for ordinal, (token_id, source, target) in enumerate(_ROUTE_SPECS)
    ),
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
_MIXED_AGENTS = _agents(
    _MIXED_POSITIONS,
    health={5: 50.0},
    included_slots=(0, 5, 9),
)
_MIXED_AGENT_MAP = {agent.global_slot: agent for agent in _MIXED_AGENTS}
_MIXED_SCENE = BattlefieldSceneV1(
    schema_version=SCENE_SCHEMA_VERSION,
    audience="researcher",
    audience_badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC FIXTURE",
    map=MapSceneV1(width=10.0, height=12.0),
    agents=_MIXED_AGENTS,
    selection=SelectionSceneV1(
        controlled_global_slot=0,
        selected_global_slot=5,
    ),
)
_MIXED_DAMAGE = _activation(
    fixture_name="mixed_net_zero",
    ordinal=0,
    token_id="basic_damage",
    source_global_slot=0,
    target_global_slot=5,
    agents=_MIXED_AGENT_MAP,
)
_MIXED_HEAL = _activation(
    fixture_name="mixed_net_zero",
    ordinal=1,
    token_id="basic_heal",
    source_global_slot=9,
    target_global_slot=5,
    agents=_MIXED_AGENT_MAP,
)
_MIXED_BATCH = _batch(
    (
        _MIXED_DAMAGE,
        _MIXED_HEAL,
        NetHealthEventV1(
            event_id="synthetic:mixed_net_zero:net-health-0",
            transition_id=1,
            recipient_global_slot=5,
            recipient_anchor=_MIXED_POSITIONS[5],
            health_before=50.0,
            health_after=50.0,
            net_delta=0.0,
            outcome="unchanged",
        ),
    ),
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
        0: (
            "stun_rogue_poison",
            "slow_rogue_poison",
            "anti_heal_rogue_poison",
        ),
        5: ("stun_hunter_trap",),
    },
    health={0: 70.0, 5: 30.0},
    class_ids=_POV_CLASS_IDS,
    included_slots=(0, 1, 5),
)
_POV_SOURCE_AGENT_MAP = {agent.global_slot: agent for agent in _POV_SOURCE_AGENTS}
_POV_SOURCE_SCENE = BattlefieldSceneV1(
    schema_version=SCENE_SCHEMA_VERSION,
    audience="researcher",
    audience_badge="PRIVILEGED RESEARCHER VIEW · SYNTHETIC SOURCE",
    map=MapSceneV1(width=10.0, height=12.0),
    agents=_POV_SOURCE_AGENTS,
    selection=SelectionSceneV1(
        controlled_global_slot=0,
        selected_global_slot=5,
    ),
    selected_legality=SelectedLegalitySceneV1(
        controlled_global_slot=0,
        target_global_slot=5,
        target_action=6,
        lane_0_available=False,
        lane_1_available=False,
        armed_lane=0,
        armed_pair_legal=False,
    ),
    pending_route=PendingRouteSceneV1(
        source_global_slot=0,
        target_global_slot=5,
        source_anchor=_MIXED_POSITIONS[0],
        target_anchor=_MIXED_POSITIONS[5],
        lane=0,
        legal=False,
    ),
    observer_visibility=(
        ObserverVisibilitySceneV1(
            observer_global_slot=0,
            candidate_global_slot=0,
            visible=True,
        ),
        ObserverVisibilitySceneV1(
            observer_global_slot=0,
            candidate_global_slot=1,
            visible=True,
        ),
        ObserverVisibilitySceneV1(
            observer_global_slot=0,
            candidate_global_slot=5,
            visible=False,
        ),
    ),
)
_POV_EXPECTED_SCENE = BattlefieldSceneV1(
    schema_version=SCENE_SCHEMA_VERSION,
    audience="agent_pov",
    audience_badge="AGENT POV · id_0 · SYNTHETIC FIXTURE",
    map=MapSceneV1(width=10.0, height=12.0),
    agents=tuple(agent for agent in _POV_SOURCE_AGENTS if agent.global_slot in (0, 1)),
    selection=SelectionSceneV1(
        controlled_global_slot=0,
        selected_global_slot=None,
    ),
)
_POV_VISIBLE_TO_HIDDEN_ACTIVATION = _activation(
    fixture_name="pov_redaction:source",
    ordinal=0,
    token_id="basic_damage",
    source_global_slot=0,
    target_global_slot=5,
    agents=_POV_SOURCE_AGENT_MAP,
)
_POV_HIDDEN_TO_VISIBLE_ACTIVATION = _activation(
    fixture_name="pov_redaction:source",
    ordinal=1,
    token_id="rogue_poison",
    source_global_slot=5,
    target_global_slot=0,
    agents=_POV_SOURCE_AGENT_MAP,
)
_POV_SOURCE_LIFECYCLE_TOKENS: tuple[StatusTokenId, ...] = (
    "stun_rogue_poison",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
)
_POV_SOURCE_LIFECYCLE_EVENTS = tuple(
    StatusLifecycleEventV1(
        event_id=f"synthetic:pov_redaction:source:status-{ordinal}",
        transition_id=1,
        recipient_global_slot=0,
        recipient_anchor=_MIXED_POSITIONS[0],
        token_id=token_id,
        change="applied",
        duration_before=0,
        duration_after=next(
            status.duration
            for status in _POV_SOURCE_AGENT_MAP[0].statuses
            if status.token_id == token_id
        ),
        source_class_id=ROGUE_CLASS_ID,
        application_event_ids=(_POV_HIDDEN_TO_VISIBLE_ACTIVATION.event_id,),
    )
    for ordinal, token_id in enumerate(_POV_SOURCE_LIFECYCLE_TOKENS)
)
_POV_SOURCE_BATCH = _batch(
    (
        _POV_VISIBLE_TO_HIDDEN_ACTIVATION,
        _POV_HIDDEN_TO_VISIBLE_ACTIVATION,
        NetHealthEventV1(
            event_id="synthetic:pov_redaction:source:net-health-0",
            transition_id=1,
            recipient_global_slot=0,
            recipient_anchor=_MIXED_POSITIONS[0],
            health_before=80.0,
            health_after=70.0,
            net_delta=-10.0,
            outcome="damage",
        ),
        NetHealthEventV1(
            event_id="synthetic:pov_redaction:source:net-health-1",
            transition_id=1,
            recipient_global_slot=5,
            recipient_anchor=_MIXED_POSITIONS[5],
            health_before=37.0,
            health_after=30.0,
            net_delta=-7.0,
            outcome="damage",
        ),
        *_POV_SOURCE_LIFECYCLE_EVENTS,
    ),
)
_POV_SAFE_ACTIVATION = AcceptedActivationEventV1(
    event_id="synthetic:pov_redaction:safe:activation-0",
    transition_id=1,
    token_id="basic_damage",
    source_global_slot=0,
    target_global_slot=None,
    source_anchor=_MIXED_POSITIONS[0],
    target_anchor=None,
    target_disclosure="redacted",
    lane=0,
    source_class_id=MAGE_CLASS_ID,
)
_POV_SAFE_LIFECYCLE_EVENTS = tuple(
    StatusLifecycleEventV1(
        event_id=f"synthetic:pov_redaction:safe:status-{ordinal}",
        transition_id=1,
        recipient_global_slot=0,
        recipient_anchor=_MIXED_POSITIONS[0],
        token_id=token_id,
        change="applied",
        duration_before=0,
        duration_after=next(
            status.duration
            for status in _POV_SOURCE_AGENT_MAP[0].statuses
            if status.token_id == token_id
        ),
        source_class_id=ROGUE_CLASS_ID,
        application_event_ids=(),
    )
    for ordinal, token_id in enumerate(_POV_SOURCE_LIFECYCLE_TOKENS)
)
_POV_SAFE_BATCH = _batch(
    (
        _POV_SAFE_ACTIVATION,
        NetHealthEventV1(
            event_id="synthetic:pov_redaction:safe:net-health-0",
            transition_id=1,
            recipient_global_slot=0,
            recipient_anchor=_MIXED_POSITIONS[0],
            health_before=80.0,
            health_after=70.0,
            net_delta=-10.0,
            outcome="damage",
        ),
        *_POV_SAFE_LIFECYCLE_EVENTS,
    )
)


_VIEWPORTS = (
    ViewportCaseV1(
        label="desktop",
        width=1440,
        height=900,
        expected_layout="split",
    ),
    ViewportCaseV1(
        label="compact",
        width=1024,
        height=768,
        expected_layout="split",
    ),
    ViewportCaseV1(
        label="minimum",
        width=960,
        height=600,
        expected_layout="split",
    ),
    ViewportCaseV1(
        label="stacked",
        width=800,
        height=900,
        expected_layout="stacked",
    ),
)
_VIEWPORT_ACTIVATIONS = (
    _activation(
        fixture_name="viewport_matrix",
        ordinal=0,
        token_id="basic_damage",
        source_global_slot=0,
        target_global_slot=5,
        agents=_CROWDED_AGENT_MAP,
    ),
    _activation(
        fixture_name="viewport_matrix",
        ordinal=1,
        token_id="hunter_trap",
        source_global_slot=2,
        target_global_slot=7,
        agents=_CROWDED_AGENT_MAP,
    ),
)
_VIEWPORT_BATCH = _batch(
    (
        *_VIEWPORT_ACTIVATIONS,
        NetHealthEventV1(
            event_id="synthetic:viewport_matrix:net-health-0",
            transition_id=1,
            recipient_global_slot=0,
            recipient_anchor=_CROWDED_POSITIONS[0],
            health_before=90.0,
            health_after=82.0,
            net_delta=-8.0,
            outcome="damage",
        ),
    )
)

RENDERER_FIXTURES: Mapping[str, RendererFixtureV1] = MappingProxyType(
    {
        "crowded_teamfight": RendererFixtureV1(
            name="crowded_teamfight",
            description=(
                "SYNTHETIC: ten adjacent agents with dense docks, auras, ranges, "
                "selection, legality, pending intent, and simultaneous events."
            ),
            scene=_CROWDED_SCENE,
            event_batch=_CROWDED_BATCH,
        ),
        "route_collision": RendererFixtureV1(
            name="route_collision",
            description=(
                "SYNTHETIC: reciprocal, parallel, crossing, and near-zero route "
                "geometry."
            ),
            scene=_ROUTE_SCENE,
            event_batch=_ROUTE_BATCH,
        ),
        "mixed_net_zero": RendererFixtureV1(
            name="mixed_net_zero",
            description=(
                "SYNTHETIC: simultaneous damage and healing intent with one exact "
                "recipient-level zero net outcome."
            ),
            scene=_MIXED_SCENE,
            event_batch=_MIXED_BATCH,
        ),
        "viewport_matrix": RendererFixtureV1(
            name="viewport_matrix",
            description=(
                "SYNTHETIC: crowded scene rendered at desktop, compact, minimum, "
                "stacked, and reduced-motion browser settings."
            ),
            scene=_CROWDED_SCENE,
            event_batch=_VIEWPORT_BATCH,
            viewports=_VIEWPORTS,
            exercise_reduced_motion=True,
        ),
        "pov_redaction": RendererFixtureV1(
            name="pov_redaction",
            description=(
                "SYNTHETIC: privileged source scene paired with an expected "
                "observer-safe payload and redacted endpoint."
            ),
            scene=_POV_EXPECTED_SCENE,
            event_batch=_POV_SAFE_BATCH,
            privileged_source_scene=_POV_SOURCE_SCENE,
            privileged_source_event_batch=_POV_SOURCE_BATCH,
        ),
    }
)


def get_renderer_fixture(name: str) -> RendererFixtureV1:
    """Return a synthetic renderer fixture by stable name."""
    try:
        return RENDERER_FIXTURES[name]
    except KeyError as exc:
        choices = ", ".join(RENDERER_FIXTURES)
        raise ValueError(
            f"unknown renderer fixture {name!r}; choose one of: {choices}."
        ) from exc


def list_renderer_fixtures() -> tuple[RendererFixtureV1, ...]:
    """Return synthetic renderer fixtures in deterministic review order."""
    return tuple(RENDERER_FIXTURES.values())
