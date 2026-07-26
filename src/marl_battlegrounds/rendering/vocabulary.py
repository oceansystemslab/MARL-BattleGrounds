"""Stable semantic tokens shared by debugger and replay presentation.

This module names presentation facts; it does not define simulator mechanics,
durations, legality, acceptance, or combat values.
"""

from dataclasses import dataclass
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

type ClassTokenId = Literal["mage", "warrior", "hunter", "rogue", "priest"]
type TeamTokenId = Literal["team_a", "team_b"]
type StatusTokenId = Literal[
    "stun_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "slow_warrior_charge",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
    "priest_freedom",
    "mage_burst",
]
type ActivationTokenId = Literal[
    "basic_damage",
    "basic_heal",
    "holy_word",
    "mage_burst",
    "warrior_charge",
    "hunter_trap",
    "rogue_poison",
]
type ModifierTokenId = Literal[
    "mage_amplification",
    "warrior_mitigation",
    "rogue_anti_heal",
    "priest_freedom",
    "mage_burst",
]
type StatusLifecycleKind = Literal[
    "applied",
    "refreshed",
    "decremented",
    "expired",
    "trap_broken",
    "cleared_unclassified",
    "trap_broken_and_reapplied",
]
type TokenFamily = Literal[
    "class",
    "team",
    "hard_control",
    "slow",
    "combat_modifier",
    "basic_activation",
    "ultimate_activation",
    "aura_modifier",
    "lifecycle",
    "unknown",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualTokenDefinition:
    """Renderer-neutral labels and fallbacks for one stable semantic token."""

    token_id: str
    label: str
    short_label: str
    accessible_name: str
    family: TokenFamily
    glyph: str
    fallback: str
    priority: int
    source_class_id: int | None

    def __post_init__(self) -> None:
        for name in (
            "token_id",
            "label",
            "short_label",
            "accessible_name",
            "glyph",
            "fallback",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                msg = f"{name} must be a non-empty Python string."
                raise ValueError(msg)
        if type(self.priority) is not int or self.priority < 0:
            msg = f"priority must be a non-negative Python int; got {self.priority!r}."
            raise ValueError(msg)
        if self.source_class_id is not None and type(self.source_class_id) is not int:
            raise ValueError("source_class_id must be a Python int or None.")


CLASS_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="mage",
        label="Mage",
        short_label="Mage",
        accessible_name="Mage class",
        family="class",
        glyph="✦",
        fallback="M",
        priority=0,
        source_class_id=MAGE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="warrior",
        label="Warrior",
        short_label="Warrior",
        accessible_name="Warrior class",
        family="class",
        glyph="◆",
        fallback="W",
        priority=1,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="hunter",
        label="Hunter",
        short_label="Hunter",
        accessible_name="Hunter class",
        family="class",
        glyph="⌖",
        fallback="H",
        priority=2,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="rogue",
        label="Rogue",
        short_label="Rogue",
        accessible_name="Rogue class",
        family="class",
        glyph="◈",
        fallback="R",
        priority=3,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="priest",
        label="Priest",
        short_label="Priest",
        accessible_name="Priest class",
        family="class",
        glyph="✚",
        fallback="P",
        priority=4,
        source_class_id=PRIEST_CLASS_ID,
    ),
)

TEAM_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="team_a",
        label="Team A",
        short_label="A",
        accessible_name="Team A, solid outline",
        family="team",
        glyph="A",
        fallback="A",
        priority=0,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="team_b",
        label="Team B",
        short_label="B",
        accessible_name="Team B, solid outline with chevron marker",
        family="team",
        glyph="B",
        fallback="B",
        priority=1,
        source_class_id=None,
    ),
)

# Hard control precedes slows, which precede combat modifiers. This is the
# canonical nine-status dock order used by every renderer.
CANONICAL_STATUS_ORDER: tuple[StatusTokenId, ...] = (
    "stun_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "slow_warrior_charge",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
    "priest_freedom",
    "mage_burst",
)

STATUS_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="stun_warrior_charge",
        label="Charge stun",
        short_label="C-STN",
        accessible_name="Warrior Charge stun",
        family="hard_control",
        glyph="⬢",
        fallback="CS",
        priority=0,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="stun_hunter_trap",
        label="Trap",
        short_label="TRAP",
        accessible_name="Hunter Trap stun",
        family="hard_control",
        glyph="▦",
        fallback="T",
        priority=1,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="stun_rogue_poison",
        label="Poison stun",
        short_label="P-STN",
        accessible_name="Rogue Poison stun",
        family="hard_control",
        glyph="⬣",
        fallback="PS",
        priority=2,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="slow_warrior_charge",
        label="Charge slow",
        short_label="C-SLW",
        accessible_name="Warrior Charge slow",
        family="slow",
        glyph="⌄",
        fallback="CS",
        priority=3,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="slow_hunter_basic",
        label="Hunter slow",
        short_label="H-SLW",
        accessible_name="Hunter Basic slow",
        family="slow",
        glyph="⌄",
        fallback="HS",
        priority=4,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="slow_rogue_poison",
        label="Poison slow",
        short_label="P-SLW",
        accessible_name="Rogue Poison slow",
        family="slow",
        glyph="⌄",
        fallback="PS",
        priority=5,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="anti_heal_rogue_poison",
        label="Anti-heal",
        short_label="ANTI",
        accessible_name="Rogue Poison anti-heal",
        family="combat_modifier",
        glyph="♡̸",
        fallback="AH",
        priority=6,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="priest_freedom",
        label="Freedom",
        short_label="FREE",
        accessible_name="Priest Blessing of Freedom",
        family="combat_modifier",
        glyph="⛓̸",
        fallback="F",
        priority=7,
        source_class_id=PRIEST_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="mage_burst",
        label="Burst",
        short_label="BURST",
        accessible_name="Mage Burst damage amplification",
        family="combat_modifier",
        glyph="✷",
        fallback="B",
        priority=8,
        source_class_id=MAGE_CLASS_ID,
    ),
)

ACTIVATION_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="basic_damage",
        label="Basic damage",
        short_label="Basic",
        accessible_name="Accepted Basic damage activation",
        family="basic_activation",
        glyph="➤",
        fallback="B",
        priority=0,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="basic_heal",
        label="Basic healing",
        short_label="Basic",
        accessible_name="Accepted Priest Basic healing activation",
        family="basic_activation",
        glyph="✚",
        fallback="B",
        priority=1,
        source_class_id=PRIEST_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="holy_word",
        label="Holy Word",
        short_label="Holy",
        accessible_name="Accepted Priest Holy Word activation",
        family="ultimate_activation",
        glyph="✥",
        fallback="U",
        priority=2,
        source_class_id=PRIEST_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="mage_burst",
        label="Burst activation",
        short_label="Burst",
        accessible_name="Accepted Mage Burst activation",
        family="ultimate_activation",
        glyph="✷",
        fallback="U",
        priority=3,
        source_class_id=MAGE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="warrior_charge",
        label="Charge",
        short_label="Charge",
        accessible_name="Accepted Warrior Charge activation",
        family="ultimate_activation",
        glyph="➠",
        fallback="U",
        priority=4,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="hunter_trap",
        label="Trap activation",
        short_label="Trap",
        accessible_name="Accepted Hunter Trap activation",
        family="ultimate_activation",
        glyph="▦",
        fallback="U",
        priority=5,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="rogue_poison",
        label="Poison",
        short_label="Poison",
        accessible_name="Accepted Rogue Poison activation",
        family="ultimate_activation",
        glyph="◆",
        fallback="U",
        priority=6,
        source_class_id=ROGUE_CLASS_ID,
    ),
)

MODIFIER_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="mage_amplification",
        label="Mage aura amplification",
        short_label="AMP",
        accessible_name="Effective Mage damage amplification aura modifier",
        family="aura_modifier",
        glyph="↑",
        fallback="AMP",
        priority=0,
        source_class_id=MAGE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="warrior_mitigation",
        label="Warrior aura mitigation",
        short_label="MIT",
        accessible_name="Effective Warrior damage mitigation aura modifier",
        family="aura_modifier",
        glyph="↓",
        fallback="MIT",
        priority=1,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="rogue_anti_heal",
        label="Anti-heal modifier",
        short_label="ANTI",
        accessible_name="Effective Rogue Poison anti-heal modifier",
        family="combat_modifier",
        glyph="♡̸",
        fallback="AH",
        priority=2,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="priest_freedom",
        label="Freedom speed floor",
        short_label="FREE",
        accessible_name="Effective Priest Freedom movement-speed floor",
        family="combat_modifier",
        glyph="⛓̸",
        fallback="F",
        priority=3,
        source_class_id=PRIEST_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="mage_burst",
        label="Burst amplification",
        short_label="BURST",
        accessible_name="Effective Mage Burst damage amplification modifier",
        family="combat_modifier",
        glyph="✷",
        fallback="B",
        priority=4,
        source_class_id=MAGE_CLASS_ID,
    ),
)

LIFECYCLE_TOKENS: tuple[VisualTokenDefinition, ...] = (
    VisualTokenDefinition(
        token_id="applied",
        label="Applied",
        short_label="Apply",
        accessible_name="Status applied",
        family="lifecycle",
        glyph="+",
        fallback="+",
        priority=0,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="refreshed",
        label="Refreshed",
        short_label="Refresh",
        accessible_name="Status refreshed or reapplied",
        family="lifecycle",
        glyph="↻",
        fallback="R",
        priority=1,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="decremented",
        label="Aged",
        short_label="Age",
        accessible_name="Status duration decremented",
        family="lifecycle",
        glyph="-",
        fallback="-",
        priority=2,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="expired",
        label="Expired",
        short_label="Expire",
        accessible_name="Status expired naturally",
        family="lifecycle",
        glyph="⌛",
        fallback="E",
        priority=3,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="trap_broken",
        label="Trap broken",
        short_label="Break",
        accessible_name="Hunter Trap ended by accepted damage",
        family="lifecycle",
        glyph="✕",
        fallback="X",
        priority=4,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="cleared_unclassified",
        label="Status ended",
        short_label="End",
        accessible_name="Status ended for an unclassified or ambiguous reason",
        family="lifecycle",
        glyph="○",
        fallback="?",
        priority=5,
        source_class_id=None,
    ),
    VisualTokenDefinition(
        token_id="trap_broken_and_reapplied",
        label="Trap broken and reapplied",
        short_label="Break+",
        accessible_name="Hunter Trap was broken and exactly reapplied",
        family="lifecycle",
        glyph="✕+",
        fallback="X+",
        priority=6,
        source_class_id=HUNTER_CLASS_ID,
    ),
)


def _lookup(
    definitions: tuple[VisualTokenDefinition, ...],
    token_id: str,
) -> VisualTokenDefinition:
    if type(token_id) is not str or not token_id.strip():
        raise ValueError("token_id must be a non-empty Python string.")
    for definition in definitions:
        if definition.token_id == token_id:
            return definition
    return VisualTokenDefinition(
        token_id=token_id,
        label="Unknown",
        short_label="?",
        accessible_name=f"Unknown visual token {token_id}",
        family="unknown",
        glyph="?",
        fallback="?",
        priority=10_000,
        source_class_id=None,
    )


def lookup_class_token(token_id: str) -> VisualTokenDefinition:
    """Return one class token definition or a safe unknown fallback."""
    return _lookup(CLASS_TOKENS, token_id)


def lookup_team_token(token_id: str) -> VisualTokenDefinition:
    """Return one team token definition or a safe unknown fallback."""
    return _lookup(TEAM_TOKENS, token_id)


def lookup_status_token(token_id: str) -> VisualTokenDefinition:
    """Return one status token definition or a safe unknown fallback."""
    return _lookup(STATUS_TOKENS, token_id)


def lookup_activation_token(token_id: str) -> VisualTokenDefinition:
    """Return one activation token definition or a safe unknown fallback."""
    return _lookup(ACTIVATION_TOKENS, token_id)


def lookup_modifier_token(token_id: str) -> VisualTokenDefinition:
    """Return one modifier token definition or a safe unknown fallback."""
    return _lookup(MODIFIER_TOKENS, token_id)


def lookup_lifecycle_token(token_id: str) -> VisualTokenDefinition:
    """Return one lifecycle token definition or a safe unknown fallback."""
    return _lookup(LIFECYCLE_TOKENS, token_id)


def class_token_from_id(class_id: int) -> VisualTokenDefinition:
    """Resolve a simulator class ID without copying class mechanics."""
    if type(class_id) is not int:
        raise ValueError("class_id must be a Python int.")
    for definition in CLASS_TOKENS:
        if definition.source_class_id == class_id:
            return definition
    return _lookup(CLASS_TOKENS, f"class_id_{class_id}")


def team_token_from_id(team_id: int) -> VisualTokenDefinition:
    """Resolve a simulator team ID without copying team mechanics."""
    if type(team_id) is not int:
        raise ValueError("team_id must be a Python int.")
    token_id = {
        TEAM_A_ID: "team_a",
        TEAM_B_ID: "team_b",
    }.get(team_id, f"team_id_{team_id}")
    return _lookup(TEAM_TOKENS, token_id)


def status_sort_key(token_id: str) -> tuple[int, str]:
    """Return the canonical status priority with deterministic unknown ordering."""
    definition = lookup_status_token(token_id)
    return definition.priority, definition.token_id


if tuple(definition.token_id for definition in STATUS_TOKENS) != (
    CANONICAL_STATUS_ORDER
):
    raise AssertionError("status-token registry order must match canonical order")
