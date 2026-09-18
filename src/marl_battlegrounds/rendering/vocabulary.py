"""Own stable visual tokens for debugger and replay presentation.

VisualTokenDefinition supplies labels, accessibility text, glyphs, fallbacks
and display priorities. Lookup helpers return known definitions or explicit
unknown fallbacks; the catalog-status mapper is strict. The version-bound IDs
remain independent of live simulator imports so recorded meanings do not drift.

These tables do not define damage, durations, action legality or acceptance.
Import-time checks enforce status-order/mapping agreement. Consumers may read
the immutable token records; importing this module draws or writes nothing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

# These are version-bound presentation vocabulary IDs, not live simulator
# imports.  Their parity with the published evaluation catalog is tested.  A
# future semantic renumbering therefore requires an explicit scene-version
# migration instead of silently changing offline rendering.
MAGE_CLASS_ID: Final = 1
WARRIOR_CLASS_ID: Final = 2
HUNTER_CLASS_ID: Final = 3
ROGUE_CLASS_ID: Final = 4
PRIEST_CLASS_ID: Final = 5
TEAM_A_ID: Final = 1
TEAM_B_ID: Final = 2

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
    """Describe one stable visual label, glyph and plain-text fallback.

    All fields are required keyword arguments. This frozen, slotted record is
    shared by debugger and replay views; it carries presentation metadata, not
    simulator mechanics. Field docs describe each value beside its declaration.

    Raises
    ------
    ValueError
        A required text field is not a nonblank Python string, priority is not
        a nonnegative Python int, or source_class_id is neither int nor None.

    Notes
    -----
    family follows TokenFamily by caller contract; construction does not check
    the Literal vocabulary. source_class_id is type-checked but not range-checked.
    """

    token_id: str
    """Stable exact ID used in serialized presentation records."""
    label: str
    """Full nonblank display label, such as a class or status name."""
    short_label: str
    """Nonblank compact label for space-limited views."""
    accessible_name: str
    """Nonblank name for assistive technology and text descriptions."""
    family: TokenFamily
    """TokenFamily category used to group presentation.

    The caller supplies a supported Literal value; construction does not
    validate this field.
    """
    glyph: str
    """Nonblank preferred display glyph, which may use Unicode."""
    fallback: str
    """Nonblank plain-text fallback when the preferred glyph is unavailable."""
    priority: int
    """Nonnegative Python int used for display ordering.

    Lower values sort first; exact ties can use token_id. Bool is not accepted.
    """
    source_class_id: int | None
    """Python int identifying the source class, or None when absent.

    Construction checks the type but does not impose a class-ID range.
    """

    def __post_init__(self) -> None:
        """Validate text and scalar metadata when a token is constructed.

        Reject blank/non-string labels, noninteger or negative priority, and a
        source class that is neither a Python int nor None with ValueError.
        The frozen record is not modified; family and class ranges are not checked.
        """
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

# Exact V1 scientific status-axis identity. Presentation ordering below never
# renumbers these channels.
CATALOG_STATUS_ID_BY_CHANNEL: Final[tuple[str, ...]] = (
    "warrior_charge_slow",
    "hunter_basic_slow",
    "rogue_poison_slow",
    "warrior_charge_stun",
    "hunter_trap_stun",
    "rogue_poison_stun",
    "rogue_poison_anti_heal",
    "mage_burst_damage_amplification",
    "priest_blessing_of_freedom_movement_floor",
)

# Evaluation records retain scientific catalog IDs and channel numbers; the
# renderer vocabulary uses stable presentation IDs.  This V2 mapping is
# explicit so neither side relies on string rewriting or channel order.
CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID: Final[Mapping[str, StatusTokenId]] = (
    MappingProxyType(
        {
            "warrior_charge_stun": "stun_warrior_charge",
            "hunter_trap_stun": "stun_hunter_trap",
            "rogue_poison_stun": "stun_rogue_poison",
            "warrior_charge_slow": "slow_warrior_charge",
            "hunter_basic_slow": "slow_hunter_basic",
            "rogue_poison_slow": "slow_rogue_poison",
            "rogue_poison_anti_heal": "anti_heal_rogue_poison",
            "priest_blessing_of_freedom_movement_floor": "priest_freedom",
            "mage_burst_damage_amplification": "mage_burst",
        }
    )
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
        label="Freezing Trap",
        short_label="FREEZE",
        accessible_name="Hunter Freezing Trap stun",
        family="hard_control",
        glyph="⬢",
        fallback="T",
        priority=1,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="stun_rogue_poison",
        label="Crippling Poison stun",
        short_label="P-STN",
        accessible_name="Rogue Crippling Poison stun",
        family="hard_control",
        glyph="⬢",
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
        glyph="↻",
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
        glyph="↻",
        fallback="HS",
        priority=4,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="slow_rogue_poison",
        label="Crippling Poison slow",
        short_label="P-SLW",
        accessible_name="Rogue Crippling Poison slow",
        family="slow",
        glyph="↻",
        fallback="PS",
        priority=5,
        source_class_id=ROGUE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="anti_heal_rogue_poison",
        label="Anti-Heal",
        short_label="ANTI",
        accessible_name="Rogue Crippling Poison Anti-Heal",
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
        label="Holy Word: Salvation",
        short_label="Salvation",
        accessible_name="Accepted Priest Holy Word: Salvation activation",
        family="ultimate_activation",
        glyph="✥",
        fallback="U",
        priority=2,
        source_class_id=PRIEST_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="mage_burst",
        label="Burst",
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
        label="Freezing Trap",
        short_label="Freezing Trap",
        accessible_name="Accepted Hunter Freezing Trap activation",
        family="ultimate_activation",
        glyph="▦",
        fallback="U",
        priority=5,
        source_class_id=HUNTER_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="rogue_poison",
        label="Crippling Poison",
        short_label="Crippling Poison",
        accessible_name="Accepted Rogue Crippling Poison activation",
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
        label="Sorcerer\u2019s Empowerment",
        short_label="AMP",
        accessible_name="Sorcerer\u2019s Empowerment damage amplification modifier",
        family="aura_modifier",
        glyph="↑",
        fallback="AMP",
        priority=0,
        source_class_id=MAGE_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="warrior_mitigation",
        label="Guardian\u2019s Barrier",
        short_label="MIT",
        accessible_name="Guardian\u2019s Barrier damage mitigation modifier",
        family="aura_modifier",
        glyph="↓",
        fallback="MIT",
        priority=1,
        source_class_id=WARRIOR_CLASS_ID,
    ),
    VisualTokenDefinition(
        token_id="rogue_anti_heal",
        label="Crippling Poison Anti-Heal",
        short_label="ANTI",
        accessible_name="Effective Rogue Crippling Poison Anti-Heal modifier",
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
        label="Apply",
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
        label="Refresh/extend",
        short_label="Refresh",
        accessible_name="Status refreshed or extended",
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
        label="Expire",
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
        label="Broken",
        short_label="Break",
        accessible_name="Freezing Trap ended by accepted damage",
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
        label="Broken, then reapplied",
        short_label="Break+",
        accessible_name="Freezing Trap was broken, then reapplied",
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
    """Find an exact token ID or create an explicit unknown-token definition.

    definitions is a tuple of immutable token records and token_id must be a
    nonblank Python string. Return the existing matching record, or a fresh
    unknown record retaining the supplied ID with priority 10,000 and '?' glyphs.
    Raise ValueError for invalid ID text; do not trim or normalize a valid ID.
    """
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
    """Return the visual definition for an exact class token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable class definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(CLASS_TOKENS, token_id)


def lookup_team_token(token_id: str) -> VisualTokenDefinition:
    """Return the visual definition for an exact team token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable team definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(TEAM_TOKENS, token_id)


def lookup_status_token(token_id: str) -> VisualTokenDefinition:
    """Return the visual definition for an exact status token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable status definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(STATUS_TOKENS, token_id)


def status_token_id_from_catalog_status_id(status_id: str) -> StatusTokenId:
    """Map a published catalog status ID to its presentation token.

    Parameters
    ----------
    status_id : str
        Exact V1 catalog status ID from CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID.

    Returns
    -------
    StatusTokenId
        The corresponding V2 presentation token string.

    Raises
    ------
    ValueError
        status_id is not a Python string or is absent from the catalog mapping.

    Notes
    -----
    This strict schema mapping has no unknown fallback. It does not consult live
    simulator constants or change status meanings in historical recordings.
    """
    if type(status_id) is not str:
        raise ValueError("status_id must be a Python string.")
    token_id = CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID.get(status_id)
    if token_id is None:
        raise ValueError(f"unknown V1 catalog status ID: {status_id!r}.")
    return token_id


def lookup_activation_token(token_id: str) -> VisualTokenDefinition:
    """Return the visual definition for an exact activation token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable activation definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(ACTIVATION_TOKENS, token_id)


def lookup_modifier_token(token_id: str) -> VisualTokenDefinition:
    """Return the visual definition for an exact modifier token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable modifier definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(MODIFIER_TOKENS, token_id)


def lookup_lifecycle_token(token_id: str) -> VisualTokenDefinition:
    """Return the visual definition for an exact lifecycle token ID.

    Parameters
    ----------
    token_id : str
        Nonblank Python token ID. Matching is exact; whitespace is not stripped.

    Returns
    -------
    VisualTokenDefinition
        The shared immutable lifecycle definition when known. Otherwise return
        a fresh unknown definition retaining token_id, with '?' glyph/fallback,
        no source class and priority 10,000.

    Raises
    ------
    ValueError
        token_id is not a Python string or is blank.

    Notes
    -----
    This lookup does not validate simulator state or alter the token registry.
    """
    return _lookup(LIFECYCLE_TOKENS, token_id)


def class_token_from_id(class_id: int) -> VisualTokenDefinition:
    """Resolve a numeric class ID to its visual definition.

    Parameters
    ----------
    class_id : int
        Python class ID. Known IDs are 1 Mage, 2 Warrior, 3 Hunter, 4 Rogue,
        and 5 Priest. Other integers are retained in an unknown fallback ID.

    Returns
    -------
    VisualTokenDefinition
        Existing immutable class token, or a fresh unknown token named
        ``class_id_<value>``. No class mechanics are copied or inferred.

    Raises
    ------
    ValueError
        class_id is not an exact Python int; bool is rejected.
    """
    if type(class_id) is not int:
        raise ValueError("class_id must be a Python int.")
    for definition in CLASS_TOKENS:
        if definition.source_class_id == class_id:
            return definition
    return _lookup(CLASS_TOKENS, f"class_id_{class_id}")


def team_token_from_id(team_id: int) -> VisualTokenDefinition:
    """Resolve a numeric team ID to its visual definition.

    Parameters
    ----------
    team_id : int
        Python team ID: 1 for Team A or 2 for Team B. Other integers use an
        unknown fallback, including the unused-slot value zero.

    Returns
    -------
    VisualTokenDefinition
        Existing immutable team token, or a fresh unknown token named
        ``team_id_<value>``.

    Raises
    ------
    ValueError
        team_id is not an exact Python int; bool is rejected.
    """
    if type(team_id) is not int:
        raise ValueError("team_id must be a Python int.")
    token_id = {
        TEAM_A_ID: "team_a",
        TEAM_B_ID: "team_b",
    }.get(team_id, f"team_id_{team_id}")
    return _lookup(TEAM_TOKENS, token_id)


def status_sort_key(token_id: str) -> tuple[int, str]:
    """Return a stable priority-and-ID sort key for a status token.

    Parameters
    ----------
    token_id : str
        Exact nonblank Python token ID. Unknown strings are allowed.

    Returns
    -------
    tuple of int and str
        Numeric priority followed by the unchanged token ID. Unknown tokens
        have priority 10,000 and sort deterministically by their ID.

    Raises
    ------
    ValueError
        token_id is not a nonblank Python string.
    """
    definition = lookup_status_token(token_id)
    return definition.priority, definition.token_id


if tuple(definition.token_id for definition in STATUS_TOKENS) != (
    CANONICAL_STATUS_ORDER
):
    raise AssertionError("status-token registry order must match canonical order")
if tuple(CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID.values()) != CANONICAL_STATUS_ORDER:
    raise AssertionError("catalog-status mapping must match canonical status order")
if set(CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID) != set(CATALOG_STATUS_ID_BY_CHANNEL):
    raise AssertionError("catalog status channel and token maps must cover one set")
