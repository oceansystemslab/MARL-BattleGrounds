"""Define scalar metric meaning, stable column order and shared display text.

MetricColumn and DirectedMetric describe the same outputs used by JAX results,
CSV files and the Viewer. Helpers supply roster-aware grouping, labels, search
facts and ordering; they compute no game metrics. All ten global slots remain
in the schema even when inactive. Run/roster identity stays separate from
numeric columns. Import builds immutable catalog tables once on the host.
Schema versions protect names/order; presentation reuses existing values.
FULL_METRIC_NAMES_BY_SCHEMA_VERSION owns the saved full-column order of every
readable current-format scalar schema (14 before the Red Zone columns, 15
now), so readers never apply new offsets to old files.
"""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Literal, cast

type MetricScope = Literal[
    "episode", "team", "agent", "team_recipient", "source_recipient", "ally_pair"
]
type MetricDirection = Literal["higher", "lower", "descriptive"]

METRIC_SCHEMA_ID = "marlbg.tdm.scalar"
METRIC_SCHEMA_VERSION = 15

# Core's terminal codes, in the same Team A/Team B win/draw/loss order as
# the priority catalog. Packing and host consistency checks share this mapping.
PRIORITY_OUTCOME_CODES = MappingProxyType(
    {
        "team_a_win": 1,
        "team_a_draw": 3,
        "team_a_loss": 2,
        "team_b_win": 2,
        "team_b_draw": 3,
        "team_b_loss": 1,
    }
)

_BURST_KILL_NOTE = (
    "A Mage's Basic attack during Burst counts as Basic and Burst help. "
    "It does not give the Mage Ultimate kill credit."
)

_KILL_HELP_TICK_RULE = (
    "Only damage and useful Priest healing on the tick the enemy dies count as "
    "help. Damage on earlier ticks does not count here."
)

_EFFECTIVE_HEALING_RULE = (
    "This healing can offset damage even if health does not rise or the ally "
    "still dies."
)

# Presentation belongs to the same catalog as scalar meaning; consumers need no
# private naming dictionary or guessed descriptions from snake_case identifiers.
METRIC_FAMILIES = MappingProxyType(
    {
        "priority": (
            "Episode Results",
            (
                "How long the game has run, each team's score, kills, deaths and "
                "rewards. Wins, draws and losses are shown when the game ends."
            ),
        ),
        "abilities": (
            "Ability Activations",
            (
                "How often agents use their Basic and Ultimate abilities, who they "
                "use them on, and how those uses are shared across targets and "
                "teammates."
            ),
        ),
        "deaths": (
            "Deaths and Time Dead",
            (
                "How often agents die and how many ticks they spend dead. Deaths "
                "after coming back to life count too."
            ),
        ),
        "kill_contributions": (
            "Kill Contributions",
            (
                "Who helped kill each enemy. Damage on the tick of the kill counts."
                " A Priest can also help by healing one of those attackers on that "
                "tick."
            ),
        ),
        "damage_done": (
            "Damage Done",
            (
                "How much damage agents deal. Includes damage beyond an enemy's "
                "remaining health."
            ),
        ),
        "recipient_damage": (
            "Damage by Recipient",
            (
                "Who dealt damage to whom. Shows team totals, agent totals and "
                "damage to each enemy, for Basic abilities, Ultimates and both "
                "together."
            ),
        ),
        "healing_done": (
            "Healing Done",
            (
                "How much Priest healing agents provide, how much goes beyond "
                "what their targets need, and how much is useful. Regeneration is "
                "separate."
            ),
        ),
        "recipient_healing": (
            "Healing by Recipient",
            (
                "Who healed whom. Shows team totals, agent totals and healing to "
                "each ally, for Basic abilities, Ultimates and both together. "
                "Separates all healing provided, useful healing and excess healing."
            ),
        ),
        "damage_received": (
            "Damage Received",
            (
                "How much damage each agent and team takes. Includes damage beyond "
                "the health they had left."
            ),
        ),
        "healing_received": (
            "Healing Received",
            (
                "Priest healing agents receive, including its useful and excess "
                "parts, plus health restored by automatic regeneration."
            ),
        ),
        "excess_healing": (
            "Excess Healing",
            (
                "Priest healing that could not fit below an ally's maximum health. "
                "Damage on the same tick is taken into account. Automatic "
                "regeneration is separate."
            ),
        ),
        "controlled_damage": (
            "Damage to Controlled Recipients",
            (
                "Damage to enemies who already had a particular status at the start"
                " of the tick. The same damage can count under more than one "
                "status."
            ),
        ),
        "controlled_healing": (
            "Healing to Controlled Recipients",
            (
                "Priest healing to allies who already had a particular harmful "
                "status at the start of the tick. The same heal can count under "
                "more than one status. Does not include regeneration."
            ),
        ),
        "controlled_kills": (
            "Kills of Controlled Recipients",
            (
                "Kills against enemies who already had a particular harmful status "
                "at the start of the tick, and who helped. One kill can count under"
                " more than one status."
            ),
        ),
        "coordination": (
            "Team Coordination",
            (
                "Whether teammates damage the same enemy, and whether each kill has"
                " one helper or several."
            ),
        ),
        "action_acceptance": (
            "Action Acceptance",
            (
                "How many chosen actions were allowed, how many were rejected, and "
                "why. Choosing to do nothing counts as allowed when that choice is "
                "legal."
            ),
        ),
        "status_applications": (
            "Status Applications",
            (
                "How often agents use abilities that apply each status, and who "
                "they use them on. Using an ability again counts once even if the "
                "status is already there."
            ),
        ),
        "status_active_steps": (
            "Time Under Status",
            (
                "How many ticks living agents spend with each status. Checks the "
                "status at the start of each tick. Team totals add everyone's time."
            ),
        ),
        "trap_breaks": (
            "Freezing Trap Breaks",
            (
                "How often enemies are trapped, which traps damage breaks, who "
                "breaks them, and how long the traps had left."
            ),
        ),
        "respawn": (
            "Respawning",
            (
                "How many respawn waves each team has had, and the average "
                "number of agents returned per wave. Also shows how many ticks "
                "agents waited on average during the recording."
            ),
        ),
        "red_zone": (
            "Red Zone",
            (
                "Kills and deaths inside each team's own Red Zone, the strip at "
                "its spawn side. Each such death counts once, but it gives the "
                "enemy team 2 points, so kills and points differ."
            ),
        ),
        "burst": (
            "Burst",
            (
                "Damage Mages deal while Burst is already active, which enemies "
                "take it, and which kills they help with."
            ),
        ),
        "aura_coverage": (
            "Aura Coverage",
            (
                "How often Mage and Warrior auras cover teammates. Shows each aura "
                "giver, each ally and the team. Multiple auras covering one ally "
                "count once in totals for each covered agent and team."
            ),
        ),
        "aura_benefits": (
            "Aura Benefits",
            (
                "Extra damage from Mage auras and damage blocked by Warrior auras. "
                "Compares the same attacks with and without the aura."
            ),
        ),
        "poison": (
            "Crippling Poison",
            (
                "Priest healing that agents miss out on because they have Rogue "
                "Poison. Does not include regeneration or guess which Rogue owns an"
                " ongoing Poison effect."
            ),
        ),
        "priest_rescue": (
            "Priest Rescue",
            (
                "Times Priest healing saves an ally from damage that would have "
                "killed them, who helped, and missed chances when the team could "
                "have saved them."
            ),
        ),
        "freedom": (
            "Freedom",
            (
                "Ticks when Freedom lets an agent move faster despite a slow. The "
                "agent does not have to move for the protection to count."
            ),
        ),
        "formation": (
            "Team Formation",
            (
                "How far apart living teammates are at the start of each tick. Each"
                " pair is measured once. The team average gives every measurement "
                "equal weight."
            ),
        ),
        "ultimate_mage": (
            "Burst (Mage Ultimate)",
            (
                "How often Mages use Burst, the damage they deal while it is "
                "active, and the kills they help with. Turning on Burst does not "
                "itself deal damage."
            ),
        ),
        "ultimate_warrior": (
            "Charge (Warrior Ultimate)",
            (
                "How often Warriors use Charge, who they charge, their damage and "
                "the kills they help with. Also shows time agents spend under "
                "Charge's slow and stun."
            ),
        ),
        "ultimate_hunter": (
            "Freezing Trap (Hunter Ultimate)",
            (
                "How often Hunters use Freezing Trap, who gets trapped, and which "
                "traps damage breaks. Breaking a Trap can free an enemy or help "
                "kill them."
            ),
        ),
        "ultimate_rogue": (
            "Crippling Poison (Rogue Ultimate)",
            (
                "How often Rogues use Crippling Poison, who they use it on, its "
                "immediate damage and the kills they help with. Also shows healing "
                "lost by poisoned agents."
            ),
        ),
        "ultimate_priest": (
            "Holy Word: Salvation (Priest Ultimate)",
            (
                "How often Priests use Salvation, who they heal, how much healing "
                "is useful or excess, and which kills or rescues they help with on "
                "that tick."
            ),
        ),
    }
)

STATUS_NAMES = (
    "warrior_charge_slow",
    "hunter_basic_slow",
    "rogue_poison_slow",
    "warrior_charge_stun",
    "hunter_trap",
    "rogue_poison_stun",
    "rogue_poison_anti_heal",
    "mage_burst",
    "priest_freedom",
)
STATUS_LABELS = (
    "Warrior Charge Slow",
    "Hunter Basic Slow",
    "Rogue Poison Slow",
    "Warrior Charge Stun",
    "Hunter Trap",
    "Rogue Poison Stun",
    "Rogue Poison Anti-Heal",
    "Mage Burst",
    "Priest Freedom",
)
STATUS_CLASS_IDS = (2, 3, 4, 2, 3, 4, 4, 1, 5)
ULTIMATE_GROUP_BY_CLASS = MappingProxyType(
    {
        1: "ultimate_mage",
        2: "ultimate_warrior",
        3: "ultimate_hunter",
        4: "ultimate_rogue",
        5: "ultimate_priest",
    }
)
_TEAMS = ((1,), (2,))
_AGENTS = tuple((slot,) for slot in range(10))
RECIPIENT_PAIRS_BY_RELATION = MappingProxyType(
    {
        "all": tuple(
            (source, recipient) for source in range(10) for recipient in range(10)
        ),
        "ally": tuple(
            (source, recipient)
            for source in range(10)
            for recipient in range(10)
            if source // 5 == recipient // 5
        ),
        "enemy": tuple(
            (source, recipient)
            for source in range(10)
            for recipient in range(10)
            if source // 5 != recipient // 5
        ),
        "self": tuple((slot, slot) for slot in range(10)),
    }
)
_ALLY_PAIRS = tuple(
    (first, second)
    for start in (0, 5)
    for first in range(start, start + 5)
    for second in range(first + 1, start + 5)
)
_RETIRED_GROUPS = {
    "burst": "ultimate_mage",
    "poison": "ultimate_rogue",
    "trap_breaks": "ultimate_hunter",
}
METRIC_GROUPS = MappingProxyType(
    {
        name: definition
        for name, definition in METRIC_FAMILIES.items()
        if name not in _RETIRED_GROUPS
    }
)


@dataclass(frozen=True, slots=True)
class MetricColumn:
    """Describe one scalar output and the evidence needed to interpret it.

    Attributes
    ----------
    name : str
        Exact exported CSV/result key, including any subject prefix.
    label : str
        Readable display label.
    family : str
        Metric family used by grouping and search.
    unit : str
        Measurement unit, such as health, count, ticks or fraction.
    scope : MetricScope
        episode, team, agent, team_recipient, source_recipient or ally_pair.
    subjects : tuple[int, ...]
        Empty for episode; configured team ID 1/2 for team; global slots
        0..9 for agent/pair roles. Team-recipient uses (team_id, global_slot).
    description : str
        Full meaning of this scalar.
    missing_when : str
        Conditions making the measurement unavailable.
    direction : MetricDirection
        higher, lower or descriptive; class context may refine guidance.
    required_class_id : int | None
        Required acting/emitting class, default None. Team scope
        requires that class among active team slots.
    status_channel : int | None
        Relevant fixed status channel, default None.
    priority : bool
        Whether this column belongs to priority output, default False.
    requires_ultimate_target : bool
        Require the declared Ultimate target relationship,
        default False.
    stem : str
        Subject-free metric key, default empty; catalog builders fill it.
    subject_role : str
        Source/recipient/contributor role, default source.
    recipient_role : str | None
        Meaning of a directed recipient, default None.
    denominator : str | None
        Meaning of the divisor for a ratio, default None.
    numerator : str | None
        Meaning of its counted amount, default None.
    requires_basic_target : bool
        Require the Basic target relationship, default False.

    Notes
    -----
    Frozen host description, not a value or validity check. The catalog builder
    owns consistent fields. A recipient's status does not identify its caster.
    """

    name: str
    label: str
    family: str
    unit: str
    scope: MetricScope
    subjects: tuple[int, ...]
    description: str
    missing_when: str
    direction: MetricDirection
    required_class_id: int | None = None
    status_channel: int | None = None
    priority: bool = False
    requires_ultimate_target: bool = False
    stem: str = ""
    subject_role: str = "source"
    recipient_role: str | None = None
    denominator: str | None = None
    numerator: str | None = None
    requires_basic_target: bool = False


def metric_guidance(
    column: MetricColumn, class_ids: tuple[int, ...] | None = None
) -> tuple[MetricDirection, str]:
    """Explain whether a scalar's direction helps its recorded team.

    Parameters
    ----------
    column : MetricColumn
        Catalog MetricColumn.
    class_ids : tuple[int, ...] | None
        Optional ten global-slot class IDs. None gives generic guidance;
        a known Warrior can make incoming damage descriptive rather than lower.

    Returns
    -------
    tuple[MetricDirection, str]
        (direction, sentence). direction is higher, lower or descriptive. The
        sentence names the relevant team and needed class/ability caveats.

    Notes
    -----
    Host-only text generation. It does not score a policy or change the metric.
    Values are interpreted in context; a larger count alone does not prove
    better tactics. Callers supply a valid ten-slot class layout.
    """
    direction = column.direction
    incoming_damage = (
        column.scope == "agent"
        and column.unit == "health"
        and column.subject_role == "recipient"
        and column.family in ("damage_received", "controlled_damage", "burst")
        and column.direction == "lower"
    )
    warrior = incoming_damage and (
        class_ids is not None and class_ids[column.subjects[0]] == 2
    )
    if warrior:
        direction = "descriptive"
    text = {
        "higher": "Higher is better",
        "lower": "Lower is better",
        "descriptive": "Context dependent",
    }[direction]
    if column.scope != "episode":
        team_id = (
            column.subjects[0]
            if column.scope in ("team", "team_recipient")
            else column.subjects[0] // 5 + 1
        )
        text += f" for Team {'A' if team_id == 1 else 'B'}"
    elif column.stem == "score_difference":
        text += " for Team A; lower is better for Team B"
    if incoming_damage and class_ids is None:
        text += "; context dependent when this agent is a Warrior"
    if warrior or (incoming_damage and class_ids is None):
        text += ". A Warrior may take hits to protect teammates"
    if column.stem.startswith("damage_to_hunter_trap_recipient"):
        text += ". Damage can free a trapped enemy"
    if column.stem == "damage_received_while_hunter_trap":
        text += ". Damage can free this agent from Trap"
    if "excess_healing" in column.stem or column.stem.startswith("excess_priest_"):
        if not column.stem.startswith("ultimate_"):
            text += (
                ". A Priest's Basic ability can still give Freedom when its "
                "healing goes beyond what the ally needs"
            )
        elif direction == "descriptive":
            text += ". This shows where excess Ultimate healing went"
        else:
            text += ". This Ultimate does not give Freedom"
    if column.stem in (
        "effective_healing_fraction",
        "basic_effective_healing_fraction",
        "effective_priest_healing_received_fraction",
    ):
        text += (
            ". A Basic cast on a full-health ally can give Freedom and lower this "
            "fraction without reducing the effective healing already given"
        )
    if column.stem in (
        "regenerated_healing",
        "regeneration_healing_received_fraction",
    ):
        text += (
            ". More regeneration can come from leaving a fight at a good time "
            "or staying away too long"
        )
    return direction, text + "."


@dataclass(frozen=True, slots=True)
class MetricTopic:
    """Describe one Viewer metric topic and its table layout.

    Attributes
    ----------
    name : str
        Stable topic key.
    label : str
        Readable topic heading.
    section : str
        Parent navigation section.
    description : str
        What a researcher can inspect here.
    paired : bool
        False for one table; True for separate totals/recipients tables.

    Notes
    -----
    Frozen host metadata; it does not duplicate the underlying scalar values.
    """

    name: str
    label: str
    section: str
    description: str
    paired: bool = False


METRIC_TOPICS = (
    MetricTopic(
        "priority",
        "Episode Results",
        "Results and Actions",
        METRIC_FAMILIES["priority"][1],
    ),
    MetricTopic(
        "abilities",
        "Ability Activations",
        "Results and Actions",
        (
            "How often agents activate their Basic and Ultimate abilities, and who "
            "they use them on. Each activation counts once."
        ),
        True,
    ),
    MetricTopic(
        "action_acceptance",
        "Accepted and Rejected Actions",
        "Results and Actions",
        "Which actions the game accepted or rejected, and why it rejected them.",
    ),
    MetricTopic(
        "damage_done",
        "Damage Done",
        "Damage and Healing",
        (
            "How much damage agents deal, including damage beyond an enemy's "
            "remaining health."
        ),
        True,
    ),
    MetricTopic(
        "damage_received",
        "Damage Received",
        "Damage and Healing",
        (
            "How much damage each agent and team takes, including damage beyond "
            "their remaining health."
        ),
    ),
    MetricTopic(
        "controlled_damage",
        "Damage to Enemies With Harmful Effects",
        "Damage and Healing",
        (
            "Damage to enemies who already had each named harmful effect at the "
            "start of the tick. Damage can count under several effects."
        ),
        True,
    ),
    MetricTopic(
        "healing_done",
        "Healing Done",
        "Damage and Healing",
        (
            "How much Priest healing agents provide, how much is useful, and how "
            "much goes beyond what their targets need. Regeneration is separate."
        ),
        True,
    ),
    MetricTopic(
        "healing_received",
        "Healing Received",
        "Damage and Healing",
        (
            "Priest healing each agent receives, its useful and excess portions, "
            "and health restored by automatic regeneration."
        ),
    ),
    MetricTopic(
        "excess_healing",
        "Excess Healing",
        "Damage and Healing",
        (
            "Priest healing that could not fit below an ally's maximum health. "
            "Damage on the same tick is taken into account."
        ),
        True,
    ),
    MetricTopic(
        "controlled_healing",
        "Healing to Allies With Harmful Effects",
        "Damage and Healing",
        (
            "Priest healing to allies who already had each named harmful effect at "
            "the start of the tick. A heal can count under several effects. "
            "Regeneration is separate."
        ),
        True,
    ),
    MetricTopic(
        "priest_rescue",
        "Priest Healing Saves",
        "Damage and Healing",
        (
            "When Priest healing saved an ally, who helped, and chances to save an "
            "ally with legal healing. Separate chances do not mean every "
            "threatened ally could be saved at once."
        ),
        True,
    ),
    MetricTopic(
        "kill_contributions",
        "Kill Contributions",
        "Kills, Deaths and Respawning",
        (
            "Who helped kill each enemy. Damage on the kill tick counts. A Priest "
            "also helps when useful healing supports an attacker on that tick."
        ),
        True,
    ),
    MetricTopic(
        "controlled_kills",
        "Kills of Enemies With Harmful Effects",
        "Kills, Deaths and Respawning",
        (
            "Kills against enemies who already had each named harmful effect at "
            "the start of the tick, and who helped. A kill can count under several "
            "effects."
        ),
        True,
    ),
    MetricTopic(
        "deaths",
        "Deaths and Time Dead",
        "Kills, Deaths and Respawning",
        (
            "How often agents die and how long they stay dead. Deaths after "
            "returning to life count too."
        ),
    ),
    MetricTopic(
        "respawn",
        "Respawning",
        "Kills, Deaths and Respawning",
        (
            "How many respawn waves each team has had, and the average "
            "number of agents returned per wave. Also shows how many ticks "
            "agents waited on average during the recording."
        ),
    ),
    MetricTopic(
        "red_zone",
        "Red Zone",
        "Kills, Deaths and Respawning",
        (
            "Kills and deaths inside each team's own Red Zone, near its spawn. "
            "Each death counts once but gives the enemy team 2 points. Shows "
            "team totals, then the agents who helped, then the agents who died."
        ),
    ),
    MetricTopic(
        "coordination",
        "Team Coordination",
        "Teamwork and Positioning",
        (
            "Whether attackers chose the same enemy, and whether one or several "
            "agents helped with each kill."
        ),
    ),
    MetricTopic(
        "formation",
        "Team Formation",
        "Teamwork and Positioning",
        (
            "How far apart living teammates are. Shows team and ally-pair "
            "averages, with the number of distance measurements."
        ),
    ),
    MetricTopic(
        "aura_coverage",
        "Aura Coverage",
        "Teamwork and Positioning",
        (
            "Which allies were close enough to receive help from Mage or Warrior "
            "auras, and for how long. Team coverage counts each ally once per tick."
        ),
        True,
    ),
    MetricTopic(
        "aura_benefits",
        "Damage Added or Blocked by Auras",
        "Teamwork and Positioning",
        (
            "Extra damage from Mage auras and damage prevented by Warrior auras. "
            "Overlapping auras do not receive invented individual shares."
        ),
    ),
    MetricTopic(
        "status_applications",
        "Status Applications",
        "Status Effects",
        (
            "How often abilities applied each named effect, and to whom. One "
            "activation can apply several different effects."
        ),
        True,
    ),
    MetricTopic(
        "status_active_steps",
        "Time With Status Effects",
        "Status Effects",
        (
            "How long each agent had each helpful or harmful effect. Effects "
            "present when recording began still count."
        ),
    ),
    MetricTopic(
        "freedom",
        "Freedom Against Slows",
        "Status Effects",
        (
            "When Freedom allowed an agent to move faster despite a slow. This "
            "measures allowed speed, not proof that the agent moved farther."
        ),
    ),
    MetricTopic(
        "ultimate_mage",
        "Burst (Mage Ultimate)",
        "Ultimates",
        (
            "Burst activations, damage while Burst was active, and help with "
            "kills. Basic attacks during Burst remain Basic attacks."
        ),
        True,
    ),
    MetricTopic(
        "ultimate_warrior",
        "Charge (Warrior Ultimate)",
        "Ultimates",
        (
            "Charge activations, damage and help with kills. Effect times belong "
            "to the agents who had the effects, including your own team."
        ),
        True,
    ),
    MetricTopic(
        "ultimate_hunter",
        "Freezing Trap (Hunter Ultimate)",
        "Ultimates",
        (
            "Freezing Trap activations, damage, help with kills, and who broke "
            "Traps. Initially trapped enemies remain visible even when the caster "
            "is unknown."
        ),
        True,
    ),
    MetricTopic(
        "ultimate_rogue",
        "Crippling Poison (Rogue Ultimate)",
        "Ultimates",
        (
            "Crippling Poison activations, damage, help with kills, harmful-effect "
            "time, and Priest healing prevented on poisoned agents."
        ),
        True,
    ),
    MetricTopic(
        "ultimate_priest",
        "Holy Word: Salvation (Priest Ultimate)",
        "Ultimates",
        (
            "Salvation activations, useful and excess healing, healing saves, and "
            "useful same-tick healing that helped attackers kill enemies."
        ),
        True,
    ),
)
METRIC_TOPICS_BY_NAME = MappingProxyType({topic.name: topic for topic in METRIC_TOPICS})
METRIC_VIEW_LABELS = MappingProxyType(
    {"totals": "Totals", "recipients": "By Recipient", "single": "Single View"}
)
_TOPIC_ORDER = {topic.name: index for index, topic in enumerate(METRIC_TOPICS)}
_TOPIC_BY_FAMILY = {
    "recipient_damage": "damage_done",
    "recipient_healing": "healing_done",
    **_RETIRED_GROUPS,
}


@dataclass(frozen=True, slots=True)
class DirectedMetric:
    """Name all scalar views of one source/recipient evidence matrix.

    Attributes
    ----------
    key : str
        Numerical matrix key and directed amount stem.
    source_total : str
        Per-source total stem.
    recipient_total : str
        Existing recipient-owned total stem.
    recipient_scope : MetricScope
        Scope used for that recipient total.
    team_total : str
        Team-level total stem.
    allocation : str | None
        Optional fraction of one source's total sent to this recipient.
    contribution : str | None
        Optional fraction of recipient/team events credited to a source.
    relation : str
        Fixed source/recipient relation used for valid pair enumeration.
    family : str
        Shared grouping/search family.
    label : str
        Base readable measurement label.
    description : str
        Base meaning used across scalar views.
    unit : str
        Unit of the amount; default health.
    subject_role : str
        Acting role; default source.
    recipient_role : str
        Receiving role; default recipient.
    required_class_id : int | None
        Required source class, default None.
    status_channel : int | None
        Relevant status channel, default None.
    requires_basic_target : bool
        Basic relationship applicability, default False.
    requires_ultimate_target : bool
        Ultimate relationship applicability, default False.
    shared_events : bool
        False for additive amounts; True when several contributors
        can share a single recipient/team event.

    Notes
    -----
    Frozen host metadata. Shared-event denominators come from unique event
    counts, not a sum of credited helpers. Existing recipient totals are reused
    instead of adding an equivalent exported column.
    """

    key: str
    source_total: str
    recipient_total: str
    recipient_scope: MetricScope
    team_total: str
    allocation: str | None
    contribution: str | None
    relation: str
    family: str
    label: str
    description: str
    unit: str = "health"
    subject_role: str = "source"
    recipient_role: str = "recipient"
    required_class_id: int | None = None
    status_channel: int | None = None
    requires_basic_target: bool = False
    requires_ultimate_target: bool = False
    shared_events: bool = False

    @property
    def amount(self) -> str:
        """Return the matrix key used as its source/recipient amount stem."""
        return self.key


def _subject_role(family: str, stem: str, scope: MetricScope) -> str:
    """Choose a column's attribution role from its actual family/stem/scope.

    Red Zone deaths and death shares belong to the agents who died (recipient).
    Red Zone kill help and kill shares belong to the helping agent
    (contributor), and team Red Zone kills to the killing team (team).
    """
    if scope == "episode":
        return "episode"
    if scope == "ally_pair":
        return "ally_pair"
    if family == "red_zone":
        if "death" in stem:
            return "recipient"
        return "contributor" if scope == "agent" else "team"
    if family in (
        "damage_received",
        "healing_received",
        "status_active_steps",
        "poison",
        "deaths",
        "respawn",
        "freedom",
    ) or stem.endswith("_received"):
        return "recipient"
    if family == "aura_coverage":
        return "emitter"
    if family in (
        "kill_contributions",
        "priest_rescue",
        "trap_breaks",
        "controlled_kills",
    ):
        return "contributor" if scope in ("agent", "source_recipient") else "team"
    return "team" if family == "priority" and scope == "team" else "source"


def _directed_metrics() -> tuple[DirectedMetric, ...]:
    """Build immutable directed-metric definitions for shared evidence matrices.

    Runs during catalog construction on the host. Definitions distinguish additive
    amounts from shared recipient events and keep fixed class/status/target roles.
    This helper does not collect numerical evidence.
    """
    metrics: list[DirectedMetric] = []

    def add(
        key: str,
        family: str,
        label: str,
        description: str,
        relation: str,
        recipient_total: str,
        *,
        source_total: str | None = None,
        team_total: str | None = None,
        allocation: str | None = None,
        contribution: str | None = None,
        recipient_scope: MetricScope = "agent",
        unit: str = "health",
        subject_role: str = "source",
        recipient_role: str = "recipient",
        required_class_id: int | None = None,
        status_channel: int | None = None,
        basic: bool = False,
        ultimate: bool = False,
        shared_events: bool = False,
        fractions: bool = True,
    ) -> None:
        """Append one directed family using explicit attribution and fraction choices.

        Required names describe matrix, grouping, relation and recipient ownership.
        Optional source/team names fall back to key; fraction stems are generated only
        when fractions=True. Flags describe applicability, not runtime event detection.
        """
        metrics.append(
            DirectedMetric(
                key=key,
                source_total=source_total or key,
                recipient_total=recipient_total,
                recipient_scope=recipient_scope,
                team_total=team_total or source_total or key,
                allocation=(allocation or f"{key}_allocation_fraction")
                if fractions
                else None,
                contribution=(contribution or f"{key}_contribution_fraction")
                if fractions
                else None,
                relation=relation,
                family=family,
                label=label,
                description=description,
                unit=unit,
                subject_role=subject_role,
                recipient_role=recipient_role,
                required_class_id=required_class_id,
                status_channel=status_channel,
                requires_basic_target=basic,
                requires_ultimate_target=ultimate,
                shared_events=shared_events,
            )
        )

    for ability in ("basic", "ultimate"):
        add(
            f"{ability}_applications",
            "abilities",
            f"{ability.title()} Ability Activations",
            "Using the ability again counts once, even if the target already has "
            "its effect and the time left does not change. An ability the game "
            "does not allow does not count.",
            "all",
            f"{ability}_applications",
            source_total=f"{ability}_activations",
            allocation=f"{ability}_application_fraction",
            contribution=f"{ability}_application_contribution_fraction",
            recipient_scope="team_recipient",
            unit="count",
            basic=ability == "basic",
            ultimate=ability == "ultimate",
        )
    for ability in ("", "basic_", "ultimate_"):
        title = ability.replace("_", " ").title()
        for effect, label, recipient, description in (
            (
                "damage_done",
                "Damage Done",
                "damage_received",
                "Counts damage after damage boosts and defenses, including "
                "damage beyond the enemy's remaining health.",
            ),
            (
                "healing_done",
                "Healing Done",
                "priest_healing_received",
                "Counts Priest healing after effects such as Poison reduce it. "
                "Includes healing that cannot fit below maximum health. "
                "Does not include regeneration.",
            ),
            (
                "effective_healing_done",
                "Effective Healing Done",
                "effective_priest_healing_received",
                "This is Priest healing minus the part that cannot fit below "
                "maximum health after that tick's damage and healing. Priests "
                "healing the same ally share the excess in proportion to how much "
                "each healed. Does not include regeneration.",
            ),
            (
                "excess_healing",
                "Excess Healing",
                "excess_healing_received",
                "Excess healing cannot fit below maximum health after "
                "that tick's damage and healing. Priests healing the same ally "
                "share the excess in proportion to how much each healed. "
                "Does not include regeneration.",
            ),
        ):
            key = ability + effect
            add(
                key,
                "recipient_damage" if effect == "damage_done" else "recipient_healing",
                title + label,
                description
                + (
                    f" Counts only {ability[:-1].title()} abilities." if ability else ""
                ),
                "enemy" if effect == "damage_done" else "ally",
                ability + recipient,
                contribution=f"{effect}_fraction"
                if not ability and effect in ("damage_done", "healing_done")
                else None,
                required_class_id=5 if effect == "excess_healing" else None,
            )
    for key, label in (
        ("burst_damage", "Damage While Burst Is Active"),
        ("burst_damage_contributing_to_kill", "Burst Damage on Enemy Death Ticks"),
    ):
        add(
            key,
            "burst",
            label,
            "The Mage must already have Burst at the start of the tick. "
            "Includes all damage while Burst is active, not just the extra damage "
            "Burst adds. Includes damage beyond the enemy's remaining health. "
            + (
                "Counts only damage on the tick the damaged enemy dies."
                if key.endswith("kill")
                else "Turning on Burst itself deals no damage."
            ),
            "enemy",
            key + "_received",
            required_class_id=1,
        )
    for prefix, source, team, participation in (
        ("", "kill_contributions", "kills", "kill_participation"),
        (
            "basic_",
            "basic_kill_contributions",
            "basic_kills",
            "basic_kill_participation",
        ),
        (
            "ultimate_",
            "ultimate_kill_contributions",
            "ultimate_kills",
            "ultimate_kill_participation",
        ),
        (
            "burst_",
            "burst_kill_contributions",
            "burst_kills",
            "burst_kill_participation",
        ),
        ("solo_", "solo_kills", "single_contributor_kills", "solo_kill_participation"),
    ):
        add(
            source,
            "burst" if prefix == "burst_" else "kill_contributions",
            (prefix.replace("_", " ").title() + "Kill Contributions")
            if prefix != "solo_"
            else "Solo Kills",
            {
                "": "Damage on the tick the enemy dies counts as help. A Priest "
                "also helps by healing one of those attackers on that tick if "
                "at least part of the heal is useful. Healing another Priest who only "
                "provided healing does not count.",
                "basic_": "Help must come from Basic damage, or Basic healing "
                "that is at least partly useful to an attacker, on the tick the "
                "enemy dies. Healing another Priest who only healed does not count.",
                "ultimate_": (
                    "The agent must activate its Ultimate on the tick the enemy dies. "
                    "It must damage the enemy, or give an attacker some useful "
                    "healing. Healing another Priest who only healed does not "
                    "count. " + _BURST_KILL_NOTE
                ),
                "burst_": "The Mage must damage the enemy on the tick it dies, "
                "with Burst already active at the start of that tick.",
                "solo_": "Only one agent may help with the kill. A Priest who "
                "gives the attacker some useful healing on that tick "
                "also counts as a helper, so that kill is not solo.",
            }[prefix],
            "enemy",
            "deaths",
            team_total=team,
            contribution=participation,
            unit="count",
            subject_role="contributor",
            recipient_role="killed_enemy",
            required_class_id=1 if prefix == "burst_" else None,
            shared_events=True,
        )
    for prefix in ("", "basic_", "ultimate_"):
        add(
            prefix + "rescue_contributions",
            "priest_rescue",
            prefix.replace("_", " ").title() + "Rescue Contributions",
            "Priest healing must keep the ally alive when that tick's damage "
            "would otherwise kill them. At least part of the helping Priest's "
            "heal must be useful. Several Priests can help; one Priest may "
            "not have enough healing to save the ally alone. "
            + (
                f"Counts only help from {prefix[:-1].title()} healing."
                if prefix
                else "Help from Basic and Ultimate healing both count."
            ),
            "ally",
            "rescues",
            team_total=prefix + "rescues",
            contribution=prefix + "rescue_participation",
            unit="count",
            subject_role="contributor",
            recipient_role="saved_ally",
            required_class_id=5 if prefix != "ultimate_" else None,
            shared_events=True,
        )
    for channel, status in enumerate(STATUS_NAMES):
        relation = "self" if channel == 7 else "ally" if channel == 8 else "enemy"
        add(
            f"{status}_applications",
            "status_applications",
            STATUS_LABELS[channel] + " Applications",
            "Using the ability again counts once, even if the target already "
            "has this status. This counts ability uses, not how often the status "
            "starts a new period. One use can give several different statuses.",
            relation,
            f"{status}_applications",
            recipient_scope="team_recipient",
            unit="count",
            required_class_id=STATUS_CLASS_IDS[channel],
            status_channel=channel,
            basic=channel in (1, 8),
            ultimate=channel not in (1, 8),
            recipient_role="affected_recipient",
        )
    for channel, status in enumerate(STATUS_NAMES[:7]):
        for effect in ("damage", "healing"):
            add(
                f"{effect}_to_{status}_recipient",
                f"controlled_{effect}",
                f"{effect.title()} to {STATUS_LABELS[channel]} Recipient",
                "The affected agent must already have this status at the start "
                "of the tick. A status first applied during that tick does not "
                "make that tick's amount count. "
                + (
                    "Includes excess Priest healing, but not regeneration."
                    if effect == "healing"
                    else "Includes damage beyond remaining health. Damage that "
                    "breaks an existing Trap still counts."
                ),
                "enemy" if effect == "damage" else "ally",
                f"{effect}_received_while_{status}",
                status_channel=channel,
                required_class_id=5 if effect == "healing" else None,
                recipient_role="affected_recipient",
            )
        add(
            f"kill_contributions_to_{status}_recipient",
            "controlled_kills",
            f"Kill Contributions Against {STATUS_LABELS[channel]}",
            (
                "The enemy must already have this status at the start of the tick. "
                "Damage on the kill tick counts as help. Healing an attacker on "
                "that tick also counts if at least part of it is useful. Healing "
                "another Priest who only healed does not count. Several agents can "
                "help with one kill."
            ),
            "enemy",
            f"deaths_while_{status}",
            team_total=f"kills_of_{status}_recipient",
            contribution=f"kill_participation_in_{status}_recipient",
            unit="count",
            subject_role="contributor",
            recipient_role="killed_enemy",
            status_channel=channel,
            shared_events=True,
        )
    add(
        "trap_break_contributions",
        "trap_breaks",
        "Trap Break Contributions",
        "The enemy must already be trapped when damage breaks the Trap. A "
        "damaging attack can help break it even if damage reductions later "
        "remove all the damage. "
        "A Trap that runs out on its own does not count.",
        "enemy",
        "trap_breaks",
        team_total="trap_breaks",
        contribution="trap_break_participation",
        unit="count",
        subject_role="contributor",
        recipient_role="trapped_enemy",
        shared_events=True,
    )
    for class_id, class_name in ((1, "mage"), (2, "warrior")):
        for measure in ("covered", "eligible"):
            add(
                f"{class_name}_aura_{measure}_steps",
                "aura_coverage",
                f"{class_name.title()} Aura {measure.title()} Steps",
                "Checks the start of each tick. The aura giver and ally must "
                "both be alive and have no spawn shield. The aura giver can "
                "count as its own ally. "
                + (
                    "The ally must be in aura range."
                    if measure == "covered"
                    else "The ally does not have to be in aura range."
                ),
                "ally",
                f"{class_name}_aura_{measure}_recipient_steps",
                unit="agent_steps",
                subject_role="emitter",
                recipient_role="covered_ally",
                required_class_id=class_id,
                shared_events=True,
                fractions=measure == "covered",
            )
    return tuple(metrics)


DIRECTED_METRICS = _directed_metrics()
ULTIMATE_TEAM_ABILITY_STEMS = MappingProxyType(
    {
        2: "warrior_charge_applications",
        4: "rogue_poison_applications",
        5: "priest_holy_word_salvation_applications",
    }
)
ULTIMATE_TEAM_ACTIVATION_STEMS = MappingProxyType(
    {
        1: "mage_burst_applications",
        3: "hunter_trap_applications",
        **ULTIMATE_TEAM_ABILITY_STEMS,
    }
)
ULTIMATE_CLASS_NAMES = ("mage", "warrior", "hunter", "rogue", "priest")


def _directed_contexts() -> MappingProxyType[
    tuple[MetricScope, str], tuple[DirectedMetric, ...]
]:
    """Index existing totals by scope/stem so display views share one scalar
    authority.
    """
    contexts: dict[tuple[MetricScope, str], list[DirectedMetric]] = {}
    for metric in DIRECTED_METRICS:
        keys: tuple[tuple[MetricScope, str], ...] = (
            ("agent", metric.source_total),
            ("team", metric.team_total),
            (metric.recipient_scope, metric.recipient_total),
        )
        for key in keys:
            contexts.setdefault(key, []).append(metric)
    return MappingProxyType({key: tuple(value) for key, value in contexts.items()})


_DIRECTED_CONTEXTS = _directed_contexts()


def _source_team_has_class(
    column: MetricColumn, class_ids: tuple[int, ...], class_id: int, relation: str
) -> bool:
    """Find the acting team, including opposite-team sources for recipient-owned
    effects.
    """
    if column.scope in ("team", "team_recipient"):
        team = column.subjects[0] - 1
    else:
        team = column.subjects[0] // 5
        if column.subject_role == "recipient" and relation == "enemy":
            team = 1 - team
    return class_id in class_ids[team * 5 : team * 5 + 5]


def metric_groups(column: MetricColumn, class_ids: tuple[int, ...]) -> tuple[str, ...]:
    """Choose relevant display groups for an existing scalar column.

    Parameters
    ----------
    column : MetricColumn
        Catalog MetricColumn whose numerical meaning stays unchanged.
    class_ids : tuple[int, ...]
        Ten global-slot class IDs for presentation; inactive slots use zero.

    Returns
    -------
    tuple[str, ...]
        Ordered unique group-name tuple. It includes the primary group and useful
        related mechanic/ability groups supported by the recorded source classes.

    Notes
    -----
    Host-only. Shared groups reuse values; they do not add counters or change
    CSV order. A healed recipient's class never stands in for its Priest
    producer, and a named status does not identify a historical caster.
    """
    primary = _RETIRED_GROUPS.get(column.family, column.family)
    if column.stem in ULTIMATE_TEAM_ABILITY_STEMS.values():
        primary = ULTIMATE_GROUP_BY_CLASS[cast(int, column.required_class_id)]
    groups = [primary]
    if column.scope == "team" and column.stem == "deaths":
        groups.append("deaths")
    if column.scope == "team" and column.stem in (
        "single_contributor_kill_fraction",
        "multi_contributor_kills",
        "multi_contributor_kill_fraction",
    ):
        groups.append("kill_contributions")
    # These all-ability counts explain the corresponding kill shares.
    # Keep their broad meaning explicit; never sum helpers to reconstruct kills.
    if (column.scope == "team" and column.stem == "kills") or (
        column.scope == "agent" and column.stem == "deaths"
    ):
        for class_id, group in ULTIMATE_GROUP_BY_CLASS.items():
            if class_id == 1 and column.scope == "team":
                continue  # Burst Totals has no share using all team kills.
            if _source_team_has_class(column, class_ids, class_id, "enemy") and (
                class_id != 5
                or any(
                    _source_team_has_class(column, class_ids, damage_class, "enemy")
                    for damage_class in (1, 2, 3, 4)
                )
            ):
                groups.append(group)
    if (
        column.stem in ("rescues", "rescue_opportunities")
        and column.scope in ("team", "agent")
        and _source_team_has_class(column, class_ids, 5, "ally")
    ):
        groups.append("ultimate_priest")
    contexts = _DIRECTED_CONTEXTS.get((column.scope, column.stem), ())
    for context in contexts:
        group = _RETIRED_GROUPS.get(context.family, context.family)
        if group in ULTIMATE_GROUP_BY_CLASS.values() and (
            context.required_class_id is not None
            and not _source_team_has_class(
                column, class_ids, context.required_class_id, context.relation
            )
        ):
            continue
        if group not in groups:
            groups.append(group)
    effect_class: int | None = None
    if column.family == "status_applications" or (
        column.stem in ULTIMATE_TEAM_ABILITY_STEMS.values()
    ):
        if column.scope == "team":
            effect_class = next(
                (
                    class_id
                    for class_id, stem in ULTIMATE_TEAM_ACTIVATION_STEMS.items()
                    if column.stem == stem
                ),
                None,
            )
    elif column.family == "status_active_steps":
        if column.status_channel not in (None, 1, 8):
            effect_class = STATUS_CLASS_IDS[column.status_channel]
    elif "ultimate_" in column.stem:
        if (
            column.scope in ("team", "team_recipient")
            and column.required_class_id is not None
        ):
            effect_class = column.required_class_id
        elif "healing" in column.stem or "rescue" in column.stem:
            if (
                column.subject_role == "recipient"
                or column.scope in ("team", "team_recipient")
                or class_ids[column.subjects[0]] == 5
            ):
                effect_class = 5
        elif (
            column.scope in ("agent", "source_recipient")
            and column.subject_role != "recipient"
        ):
            effect_class = class_ids[column.subjects[0]]
    # Class views show effects that ability can produce. General tables retain
    # their valid zero values, including Mage activation's lack of direct damage.
    if (
        effect_class == 1
        and any(
            effect in column.stem for effect in ("damage", "healing", "kill", "rescue")
        )
    ) or (effect_class == 5 and "damage" in column.stem):
        effect_class = None
    if effect_class is not None and (
        column.scope == "team_recipient"
        or (column.scope == "team" and column.subject_role != "recipient")
        or (column.subject_role == "recipient" and "healing" in column.stem)
    ):
        relation = contexts[0].relation if contexts else "ally"
        if not _source_team_has_class(column, class_ids, effect_class, relation):
            effect_class = None
    if effect_class is not None and effect_class in ULTIMATE_GROUP_BY_CLASS:
        group = ULTIMATE_GROUP_BY_CLASS[effect_class]
        if group not in groups:
            groups.append(group)
    return tuple(dict.fromkeys(groups))


def metric_view_order(column: MetricColumn) -> int:
    """Return an ability-detail sorting category for a catalog column.

    Parameters
    ----------
    column : MetricColumn
        Existing MetricColumn.

    Returns
    -------
    int
        Integer 0 for activation/application, 1 for amounts, 2 for events such as
        kills/rescues/deaths, or 3 for other timing/control details.

    Notes
    -----
    Host-only display ordering; the exported CSV order is unchanged.
    """
    if "activation" in column.stem or "application" in column.stem:
        return 0
    if column.family in ("status_active_steps", "trap_breaks", "poison"):
        return 3
    if any(event in column.stem for event in ("kill", "rescue", "death")):
        return 2
    if column.unit == "health" or any(
        effect in column.stem for effect in ("damage", "healing")
    ):
        return 1
    return 3


def metric_primary_location(column: MetricColumn) -> tuple[str, str]:
    """Give a scalar column one fixed catalog home.

    Parameters
    ----------
    column : MetricColumn
        Existing MetricColumn with a known family/stem.

    Returns
    -------
    tuple[str, str]
        (topic_name, view_name), where view is single, totals or recipients.

    Notes
    -----
    Host-only and independent of the current roster. Shared display locations
    may add context elsewhere, but this fixed home controls canonical ordering.
    """
    if column.priority:
        topic = "priority"
    elif column.stem in ULTIMATE_TEAM_ABILITY_STEMS.values():
        topic = ULTIMATE_GROUP_BY_CLASS[cast(int, column.required_class_id)]
    elif column.scope == "team" and column.family == "healing_received":
        topic = "healing_received"
    elif "excess" in column.stem:
        topic = "excess_healing"
    elif column.stem.startswith(("single_contributor_", "multi_contributor_")):
        topic = "kill_contributions"
    else:
        topic = _TOPIC_BY_FAMILY.get(column.family, column.family)
    return topic, _metric_view(column, topic)


def _metric_view(column: MetricColumn, topic: str) -> str:
    """Select single/totals/recipients for a known topic without duplicating sibling
    rows.
    """
    if not METRIC_TOPICS_BY_NAME[topic].paired:
        return "single"
    if column.scope in ("team_recipient", "source_recipient") or (
        column.subject_role == "recipient"
        and (column.scope == "agent" or topic in ("healing_done", "excess_healing"))
    ):
        return "recipients"
    return "totals"


def metric_locations(
    column: MetricColumn, class_ids: tuple[int, ...]
) -> tuple[tuple[str, str], ...]:
    """List the tables that can reuse an existing scalar in this roster.

    Parameters
    ----------
    column : MetricColumn
        Catalog MetricColumn.
    class_ids : tuple[int, ...]
        Ten global-slot class IDs, with zero for inactive slots.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Tuple of (topic, view) pairs in fixed topic order. The column's canonical
        home is included; additional tables share the same numerical value.

    Notes
    -----
    Host-only. Context can add useful ability/recipient views, but no scalar is
    copied into both sibling tables or assigned a new meaning.
    """
    topics = {
        _TOPIC_BY_FAMILY.get(group, group) for group in metric_groups(column, class_ids)
    }
    topics.add(metric_primary_location(column)[0])
    if column.scope == "team" and column.family == "healing_received":
        topics.difference_update(("excess_healing", "healing_done"))
    elif "excess" in column.stem:
        topics.update(("excess_healing", "healing_done"))
        if column.subject_role == "recipient":
            topics.add("healing_received")
    return tuple(
        (topic, _metric_view(column, topic))
        for topic in sorted(topics, key=_TOPIC_ORDER.__getitem__)
    )


_ULTIMATE_NAMES = ("Burst", "Charge", "Freezing Trap", "Crippling Poison", "Salvation")
_ULTIMATE_TOPIC_NAMES = (
    "Mage Burst",
    "Warrior Charge",
    "Hunter Trap",
    "Rogue Poison",
    "Priest Salvation",
)


def metric_topic_text(
    column: MetricColumn, topic: str, class_ids: tuple[int, ...]
) -> dict[str, str]:
    """Specialize a shared column's wording for a named Ultimate view.

    Parameters
    ----------
    column : MetricColumn
        Existing MetricColumn.
    topic : str
        Topic key. Non-Ultimate topics return no overrides.
    class_ids : tuple[int, ...]
        Ten global-slot class IDs used to identify actual source ability.

    Returns
    -------
    dict[str, str]
        Dict of changed text fields only: labels, description, guidance, ratio
        wording or subtitle. {} means the ordinary catalog text remains sufficient.

    Notes
    -----
    Host-only; values, validity and CSV identity stay unchanged. Wording keeps
    combined team denominators explicit and never treats a recipient's class
    as evidence that it produced the effect.
    """
    if topic not in ULTIMATE_GROUP_BY_CLASS.values():
        return {}
    class_id = next(k for k, value in ULTIMATE_GROUP_BY_CLASS.items() if value == topic)
    ability = _ULTIMATE_TOPIC_NAMES[class_id - 1]
    stem = column.stem
    team = (
        "Team A"
        if (
            column.subjects[0] == 1
            if column.scope in ("team", "team_recipient")
            else column.subjects[0] < 5
        )
        else "Team B"
    )
    description = column.description
    if column.scope == "team" and stem == ULTIMATE_TEAM_ACTIVATION_STEMS.get(class_id):
        description = (
            f"How many times {team}'s {ULTIMATE_CLASS_NAMES[class_id - 1].title()}s "
            f"activated {ability}. "
            "Each activation counts once, even if it applies several effects."
        )
    elif stem in ("ultimate_activations", "ultimate_applications"):
        target = " on this target" if column.scope == "source_recipient" else ""
        description = f"How many times this agent activated {ability}{target}. "
        description += "Each activation counts once. A rejected action does not count."
        if class_id == 1:
            description += (
                " Burst targets the Mage itself and deals no immediate damage."
            )
    elif stem == "ultimate_damage_done":
        target = " to this enemy" if column.scope == "source_recipient" else ""
        description = (
            f"How much damage this agent dealt{target} with {ability}. Includes "
            "damage boosts, defenses and damage beyond the enemy's remaining health."
        )
    elif stem == "kills" and column.scope == "team":
        description += " Includes all abilities. Used by the team kill shares here."
    elif stem == "deaths" and column.scope == "agent":
        description += (
            " Includes all abilities. Used by the shares of this enemy's deaths "
            "in the source-to-recipient rows."
        )
    elif stem == "rescues":
        description += (
            " Includes Basic and Ultimate healing. Used by the healing-save "
            "shares here. Each saved ally counts once per tick."
        )
    elif stem == "rescue_opportunities":
        description += (
            " Includes available Basic and Ultimate healing, not just Salvation."
        )
    elif stem.startswith("ultimate_kill_") and column.scope in (
        "agent",
        "source_recipient",
    ):
        # The source may move to another slot, but it must still match this view.
        if class_ids[column.subjects[0]] != class_id:
            return {}
        recipient = column.scope == "source_recipient"
        enemy = "this enemy" if recipient else "enemies"
        rule = (
            f"This Priest must give useful {ability} healing to an ally who "
            "damages that enemy on its death tick. Entirely excess "
            "healing does not count."
            if class_id == 5
            else f"This agent must damage the enemy with {ability} "
            "on the tick the enemy dies."
        )
        if stem == "ultimate_kill_contributions":
            description = f"How many times this agent helped kill {enemy}. " + rule
        elif stem == "ultimate_kill_participation":
            description = (
                "What share of "
                + ("this enemy's deaths" if recipient else team + "'s kills")
                + " "
                f"this agent helped with using {ability}. Includes all abilities "
                "in the death or kill total. " + rule
            )
        else:
            description = (
                f"What share of this agent's kills helped by {ability} involved "
                "this enemy. " + rule
            )
        description += (
            " Count this helper once per enemy death. Other agents can help with "
            "the same death. This does not prove the ability was needed for the kill."
        )
    original = {
        "label": column.label,
        "description": column.description,
        "numerator": column.numerator,
        "denominator": column.denominator,
        "guidance": metric_guidance(column, class_ids)[1],
        "missing_when": column.missing_when,
    }
    text = {**original, "description": description}
    if stem == "ultimate_activations" or (
        column.scope == "team" and stem == ULTIMATE_TEAM_ACTIVATION_STEMS.get(class_id)
    ):
        text["label"] = f"{ability} Applications"
    elif stem == "ultimate_applications":
        text["label"] = f"{ability} Applications on Recipient"

    # These shared team totals include other classes. Name the source's ability
    # separately from the combined count used to divide it.
    if stem == "ultimate_application_contribution_fraction":
        text["label"] = (
            f"{ability} Share of {team}'s Combined Applications on Recipient"
        )
        text["description"] = (
            f"What share of {team}'s combined ability applications on this target "
            f"came from this agent's {ability}. The team total includes "
            "Mage Burst, Warrior Charge, Hunter Trap, Rogue Poison and Priest "
            "Salvation. Each allowed use counts once. For example, 4 of the "
            "team's 8 applications gives 4/8 = 0.5 (50%)."
        )
        text["denominator"] = (
            f"All applications by {team} on this target across Mage Burst, "
            "Warrior Charge, Hunter Trap, Rogue Poison and Priest Salvation"
        )
    elif stem == "ultimate_damage_done_contribution_fraction":
        text["label"] = (
            f"{ability} Share of {team}'s Combined Ability Damage to Recipient"
        )
        text["description"] = (
            f"What share of {team}'s combined ability damage to this target came "
            f"from this agent's {ability}. The team total includes direct damage "
            "from Warrior Charge, Hunter Trap and Rogue Poison. Counts damage "
            "after damage boosts and defenses, including damage beyond the "
            "enemy's remaining health."
        )
        text["denominator"] = (
            f"All direct Warrior Charge, Hunter Trap and Rogue Poison damage "
            f"{team} dealt this target, including this agent's damage"
        )
    if column.denominator and text["denominator"] != column.denominator:
        denominator = cast(str, text["denominator"])
        for old, new in (
            (column.denominator, denominator),
            (
                column.denominator[:1].lower() + column.denominator[1:],
                denominator[:1].lower() + denominator[1:],
            ),
        ):
            text["missing_when"] = cast(str, text["missing_when"]).replace(old, new)

    def name_ability(value: str) -> str:
        # Received healing already names Priest; keep that class name once.
        """Name the actual Ultimate consistently without repeating a class name."""
        value = value.replace("Ultimate Effective Priest", "Effective Priest Salvation")
        value = value.replace("Ultimate Priest", "Priest Salvation")
        value = re.sub(r"\ban Ultimate\b", ability, value)
        value = re.sub(r"\bUltimates?\b(?: [Aa]bilit(?:y|ies))?", ability, value)
        for pattern, name in zip(
            (
                r"\b(?:Mage )?Burst\b",
                r"\b(?:Warrior )?Charge\b",
                r"\b(?:Hunter )?(?:Freezing )?Trap\b",
                r"\b(?:Rogue )?(?:Crippling )?Poison\b",
                r"\b(?:Priest )?Salvation\b",
            ),
            _ULTIMATE_TOPIC_NAMES,
            strict=True,
        ):
            value = re.sub(pattern, name, value)
        return value

    result = {
        field: named
        for field, value in text.items()
        if value is not None and (named := name_ability(value)) != original[field]
    }
    if stem == "ultimate_activations" or (
        column.scope == "team" and stem == ULTIMATE_TEAM_ACTIVATION_STEMS.get(class_id)
    ):
        result["subtitle"] = ""
    return result


def _row_order(
    column: MetricColumn, topic: str, measure_order: dict[str, tuple[int, int]]
) -> tuple[int, ...]:
    """Sort subjects and related measures using stable catalog keys rather than display
    labels.
    """
    if column.scope in ("episode", "team"):
        section = 0
    elif column.scope == "team_recipient" or column.subject_role == "recipient":
        section = 2
    elif column.scope == "source_recipient":
        section = 3
    elif column.scope == "ally_pair":
        section = 4
    else:
        section = 1
    ability = (
        0
        if column.status_channel is not None
        else 1
        if column.stem.startswith("basic_") or "_basic_kill" in column.stem
        else 2
        if "ultimate_" in column.stem
        else 0
    )
    status_order = (1, 8, 7, 0, 3, 4, 2, 5, 6)
    return (
        section,
        column.subjects[0] if column.subjects else -1,
        column.subjects[1] if len(column.subjects) > 1 else -1,
        metric_view_order(column) if topic.startswith("ultimate_") else 0,
        ability,
        status_order.index(column.status_channel)
        if column.status_channel is not None
        else 0,
        *measure_order[column.stem],
    )


def metric_order_key(column: MetricColumn, topic: str) -> tuple[int, ...]:
    """Return the stable row-order key for a metric within a topic.

    Parameters
    ----------
    column : MetricColumn
        Existing catalog column.
    topic : str
        Known display topic controlling any Ultimate-detail ordering.

    Returns
    -------
    tuple[int, ...]
        Integer tuple ordering scope, subjects, ability/status and related measures.

    Notes
    -----
    Host-only. Shared rows keep the same subject/measure order across tables;
    no numerical data or mutable display state is read.
    """
    return _row_order(column, topic, _MEASURE_ORDER)


def metric_csv_order(column: MetricColumn) -> tuple[int, ...]:
    """Return a column's fixed topic/table/measurement sort key.

    Parameters
    ----------
    column : MetricColumn
        Existing catalog column.

    Returns
    -------
    tuple[int, ...]
        Integer tuple based on its canonical home and row-order key.

    Notes
    -----
    Host-only; consumers use this while building the full schema, preserving
    the separately declared priority prefix. This does not sort runtime rows.
    """
    topic, view = metric_primary_location(column)
    return (
        _TOPIC_ORDER[topic],
        int(view == "recipients"),
        *metric_order_key(column, topic),
    )


def _prefix(scope: MetricScope, subjects: tuple[int, ...]) -> str:
    """Build the stable CSV subject prefix from scope and configured team/global
    slots.
    """
    if scope == "episode":
        return ""
    if scope == "team":
        return "team_a_" if subjects[0] == 1 else "team_b_"
    if scope == "team_recipient":
        team = "a" if subjects[0] == 1 else "b"
        return f"team_{team}_to_agent_{subjects[1]}_"
    if scope == "source_recipient":
        return f"agent_{subjects[0]}_to_agent_{subjects[1]}_"
    if scope == "ally_pair":
        return f"agent_{subjects[0]}_and_agent_{subjects[1]}_"
    return f"agent_{subjects[0]}_"


def _allocation_label(metric: DirectedMetric) -> str:
    """Describe how much of one source's total went to the named recipient."""
    ability = (
        "Basic "
        if metric.key.startswith("basic_")
        else "Ultimate "
        if metric.key.startswith("ultimate_")
        else ""
    )
    if metric.family in ("recipient_damage", "recipient_healing"):
        amount = (
            "Damage"
            if metric.family == "recipient_damage"
            else "Effective Healing"
            if "effective" in metric.key
            else "Excess Healing"
            if "excess" in metric.key
            else "Healing"
        )
        target = "Enemy" if metric.relation == "enemy" else "Ally"
        return f"Share of This Agent's {ability}{amount} That Went to This {target}"
    if metric.family == "burst" and not metric.shared_events:
        detail = " on Enemy Death Ticks" if metric.key.endswith("kill") else ""
        return f"Share of This Mage's Burst Damage{detail} That Went to This Enemy"
    if metric.family == "status_applications":
        return (
            f"Share of This Agent's {STATUS_LABELS[metric.status_channel or 0]} "
            "Applications on This Target"
        )
    if metric.family == "aura_coverage":
        aura = "Mage" if metric.required_class_id == 1 else "Warrior"
        return f"Share of This {aura}'s Aura Coverage That Went to This Ally"
    if metric.family in ("controlled_damage", "controlled_healing"):
        amount = "Damage" if metric.family == "controlled_damage" else "Healing"
        return (
            f"Share of This Agent's {amount} to Targets With This Effect "
            "That Went to This Target"
        )
    if metric.family == "priest_rescue":
        return f"Share of This Priest's {ability}Saves That Involved This Ally"
    if metric.family == "trap_breaks":
        return "Share of This Agent's Trap Break Contributions Against This Enemy"
    if metric.key == "solo_kills":
        return "Share of This Agent's Solo Kills Against This Enemy"
    if metric.family in ("kill_contributions", "controlled_kills", "burst"):
        condition = (
            " Against Enemies With This Effect"
            if metric.status_channel is not None
            else " During Burst"
            if metric.family == "burst"
            else ""
        )
        return (
            f"Share of This Agent's {ability}Kill Contributions{condition} "
            "That Involved This Enemy"
        )
    raise ValueError(f"Missing allocation label for {metric.key}")


def _directed_description(
    metric: DirectedMetric, scope: MetricScope, *, recipient: bool = False
) -> str:
    """Build role-correct wording for one directed amount/fraction view."""
    pair = scope == "source_recipient"
    rules = metric.description
    if recipient:
        rules = rules.replace(
            "the enemy's remaining health", "this agent's remaining health"
        ).replace("the damaged enemy dies", "this agent dies")
    actor = "this team" if scope == "team" else "this agent"
    target = "this enemy" if metric.relation == "enemy" else "this ally"
    if metric.relation in ("all", "self"):
        target = "this target"
    if metric.family == "abilities":
        ability = metric.key.removesuffix("_applications").title()
        if recipient:
            return (
                f"How many times agents on this team used their {ability} "
                "abilities on this target. Each use counts once."
            )
        return (
            f"How many times {actor} used its {ability} ability on {target}. " + rules
        )
    if metric.family == "status_applications":
        actor = "agents on this team" if recipient else actor
        return (
            f"How many times {actor} used the ability that gives this status "
            f"on {target}. " + rules
        )
    if metric.family == "aura_coverage":
        aura = "Mage" if metric.required_class_id == 1 else "Warrior"
        covered = "covered" in metric.key
        if recipient:
            lead = (
                f"How many ticks at least one {aura} aura covered this agent. "
                if covered
                else f"How many ticks this agent could receive a {aura} aura, "
                "even if it was too far away. "
            )
            return lead + (
                "Checks the start of the tick. This agent and at least one "
                "aura giver must be alive and have no spawn shield. Counts "
                "this agent once per tick even with several aura givers. "
                "An aura giver can cover itself."
            )
        verb = "covered" if covered else "could cover"
        return f"How many ticks {actor}'s aura {verb} {target}. " + rules
    if metric.shared_events:
        if recipient:
            if metric.recipient_total == "deaths":
                return (
                    "How many times this agent died, whoever helped and "
                    "whichever ability they used. All deaths count, not only "
                    "deaths involving the ability shown in this group."
                )
            if metric.recipient_total == "rescues":
                return (
                    "How many times Priest healing saved this agent from "
                    "damage that would otherwise kill it. Counts each rescue "
                    "once, whoever helped and whichever healing ability they used."
                )
            if metric.family == "controlled_kills":
                return (
                    "How many times this agent died while it already had this "
                    "status at the start of the tick. Each death counts once "
                    "for this status."
                )
            return (
                "How many times damage broke a Trap on this agent. Each break "
                "counts once even if several attackers helped. A Trap that "
                "ran out on its own does not count."
            )
        if metric.family == "priest_rescue":
            event = "rescue"
            target = target if pair else "allies"
            lead = f"How many times {actor} helped save {target} from death. "
        elif metric.family == "trap_breaks":
            event = "Trap break"
            trapped = "this enemy's Trap" if pair else "enemy traps"
            lead = f"How many times {actor} helped break {trapped} with damage. "
        else:
            event = "kill"
            target = target if pair else "enemies"
            lead = f"How many times {actor} helped kill {target}. "
        if scope == "team":
            lead += f"Each {event} counts once even if several teammates helped. "
        elif metric.key != "solo_kills" and metric.family != "controlled_kills":
            lead += (
                f"Each agent counts once per {event}. Several agents can help "
                f"with the same {event}. "
            )
        return lead + rules
    if metric.family in ("controlled_damage", "controlled_healing"):
        effect = "damage" if metric.family == "controlled_damage" else "healing"
        verb = "dealt" if effect == "damage" else "gave"
        lead = (
            f"How much {effect} this agent received while it had this status. "
            if recipient
            else f"How much {effect} {actor} {verb} to {target} while the target "
            "had this status. "
        )
        return lead + rules
    if metric.family == "burst":
        lead = (
            "How much damage this agent took from Mages with Burst active. "
            if recipient
            else f"How much damage {actor} dealt to {target} with Burst active. "
        )
        return lead + rules
    channel = (
        "Basic "
        if metric.key.startswith("basic_")
        else "Ultimate "
        if metric.key.startswith("ultimate_")
        else ""
    )
    destination = f" to {target}" if pair else ""
    if "damage" in metric.key:
        lead = (
            f"How much {channel}damage this agent took. "
            if recipient
            else f"How much {channel}damage {actor} dealt{destination}. "
        )
    elif "excess" in metric.key:
        lead = (
            f"How much {channel}healing this agent received that was more than needed. "
            if recipient
            else f"How much of {actor}'s {channel}healing{destination} "
            "was more than needed. "
        )
    else:
        useful = " that was useful" if "effective" in metric.key else ""
        lead = (
            f"How much {channel}Priest healing this agent received{useful}. "
            if recipient
            else f"How much {channel}Priest healing {actor} gave{destination}{useful}. "
        )
    return lead + rules


@cache
def _numerator(stem: str, scope: MetricScope, status_channel: int | None) -> str:
    """Describe the counted quantity for a known ratio stem and optional status
    channel.
    """
    subject = "this team" if scope == "team" else "this agent"
    destination = " to this target" if scope == "source_recipient" else ""
    quantities = {
        "death_fraction": "this agent's deaths",
        "dead_step_fraction": "ticks this agent spent dead",
        "kill_contributions_per_death": "kills this agent helped with",
        "kill_participation": "kills this agent helped with",
        "basic_kill_participation": (
            "kills this agent helped with using its Basic ability"
        ),
        "basic_kill_fraction": "kills this team got with help from a Basic ability",
        "solo_kill_fraction": "kills this agent got without another agent helping",
        "damage_done_fraction": f"damage this agent dealt{destination}",
        "healing_done_fraction": f"Priest healing this agent gave{destination}",
        "damage_received_fraction": "damage this agent took",
        "healing_received_fraction": (
            "Priest healing this agent received, including excess, plus health "
            "restored by regeneration"
        ),
        "priest_healing_received_fraction": (
            f"Priest healing {subject} received, including excess"
        ),
        "regeneration_healing_received_fraction": (
            f"health {subject} got back from regeneration"
        ),
        "single_contributor_kill_fraction": (
            "kills this team got with exactly one helper"
        ),
        "multi_contributor_kill_fraction": (
            "kills this team got with two or more helpers"
        ),
        "focus_fire_concentration": (
            "the largest share of damaging teammates on this team attacking "
            "the same enemy on each qualifying tick, added together"
        ),
        "action_acceptance_rate": (
            "actions sent by agents on this team where every choice was allowed"
            if scope == ("team")
            else ("actions this agent sent where every choice was allowed")
        ),
        "trap_break_rate": "enemy traps this team broke with damage",
        "trap_mean_remaining_steps_at_break": (
            "ticks left on each enemy Trap this team broke, added together, "
            "after that tick's timer decrease"
        ),
        "trap_break_participation": "enemy traps this agent helped break with damage",
        "mean_observed_respawn_wait_steps": (
            "ticks each agent on this team spent dead during the recording, "
            "added together"
        ),
        "mean_agents_per_respawn_wave": (
            "times agents were brought back by this team's respawn waves"
        ),
        "burst_damage_fraction": (
            "damage this team's Mages dealt with Burst already active at the "
            "start of the tick"
        )
        if scope == "team"
        else (
            "damage this Mage dealt with Burst already active at the start of the tick"
        ),
        "rescue_rate": "times this team saved an ally from death with Priest healing",
        "rescue_participation": "rescues this Priest helped with",
        "freedom_protection_fraction": (
            "ticks when Freedom reduced a slow on each agent on this team, "
            "added together"
        )
        if scope == "team"
        else "ticks when Freedom reduced a slow on this agent",
        "ally_distance_mean": (
            "distances between living teammates on this team, added together"
            if scope == "team"
            else (
                "distances between these two teammates while both were alive, "
                "added together"
            )
        ),
    }
    if stem in quantities:
        return quantities[stem]
    for ability in ("basic", "ultimate"):
        if stem == f"{ability}_application_fraction":
            return f"times this agent used its {ability.title()} ability on this target"
    for outcome in ("kill", "rescue"):
        if stem == f"ultimate_{outcome}_participation":
            return f"{outcome}s this agent helped with using its Ultimate"
        if stem == f"ultimate_{outcome}_fraction":
            return f"{outcome}s this team got with help from an Ultimate"
    for class_name in ULTIMATE_CLASS_NAMES:
        if stem == f"{class_name}_basic_kill_fraction":
            return (
                f"kills this team's {class_name.title()}s helped with using "
                "their Basic abilities, counting each enemy death once"
            )
    for class_name in ULTIMATE_CLASS_NAMES[1:]:
        if stem == f"{class_name}_ultimate_kill_fraction":
            return (
                "kills this team's "
                f"{class_name.title()}"
                "s helped with using their Ultimates, counting each enemy death "
                "once"
            )
    if status_channel is not None and stem == (
        f"kill_participation_in_{STATUS_NAMES[status_channel]}_recipient"
    ):
        return (
            "kills this agent helped with against enemies who already had "
            "this status at the start of the tick"
        )
    for ability in ("", "basic_", "ultimate_"):
        title = ability.replace("_", " ").title()
        for portion in ("effective", "excess"):
            useful = "was useful" if portion == "effective" else "was more than needed"
            if stem == f"{ability}{portion}_healing_fraction":
                return f"{title}healing {subject} gave{destination} that {useful}"
            if not ability and stem == f"{portion}_priest_healing_received_fraction":
                return f"Priest healing {subject} received that {useful}"
    for aura in ("mage", "warrior"):
        if stem == f"{aura}_aura_coverage":
            if scope == "team":
                return (
                    "ticks agents on this team were covered by a "
                    f"{aura.title()}"
                    " aura, added together; each agent counts once per tick"
                )
            if scope == "source_recipient":
                return f"ticks this {aura.title()}'s aura covered this ally"
            return (
                "ticks this "
                f"{aura.title()}"
                "'s aura covered allies, added together; each covered ally counts "
                "once per tick"
            )
    raise ValueError(f"missing numerator description: {stem} ({scope})")


def _team_words(text: str, scope: MetricScope, subjects: tuple[int, ...]) -> str:
    """Replace generic subject wording with the fixed team named by this column."""
    text = text.strip()
    text = text[:1].upper() + text[1:]
    if scope == "episode":
        return text
    team = subjects[0] - 1 if scope in ("team", "team_recipient") else subjects[0] // 5
    name = "Team A" if team == 0 else "Team B"
    text = re.sub(
        ("\\b(?:this|the(?: whole)?|its|their) team\\b"),
        name,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\bTeam\b(?! [AB]\b)", name, text).strip()


def _measure_order(columns: list[MetricColumn]) -> dict[str, tuple[int, int]]:
    """Group each amount/count beside its related fractions/rates using stable stems."""
    order: dict[str, int] = {}

    def place(*stems: str | None) -> None:
        """Assign the next position once per non-None stem while keeping first-use
        order.
        """
        for stem in stems:
            if stem is not None and stem not in order:
                order[stem] = len(order)

    # The directed catalog already declares All, Basic, then Ultimate effects.
    for metric in DIRECTED_METRICS:
        place(
            metric.source_total,
            metric.team_total,
            metric.recipient_total,
            metric.amount,
        )
        place(metric.allocation, metric.contribution)
        if metric.key.endswith("effective_healing_done"):
            place(metric.key.removesuffix("_done") + "_fraction")
            place(metric.recipient_total + "_fraction")
        elif metric.key.endswith("excess_healing"):
            place(metric.key + "_fraction")
        if metric.key == "kill_contributions":
            place("kill_contributions_per_death")
        elif metric.key == "solo_kills":
            place("solo_kill_fraction")
        elif metric.key == "basic_kill_contributions":
            place("basic_kill_fraction")
            for name in ULTIMATE_CLASS_NAMES:
                place(name + "_basic_kills", name + "_basic_kill_fraction")
        elif metric.key == "ultimate_kill_contributions":
            place("ultimate_kill_fraction")
            for name in ULTIMATE_CLASS_NAMES[1:]:
                place(name + "_ultimate_kills", name + "_ultimate_kill_fraction")
        elif metric.key == "rescue_contributions":
            place("rescue_rate")
        elif metric.key == "basic_rescue_contributions":
            place("basic_rescue_fraction")
        elif metric.key == "ultimate_rescue_contributions":
            place("ultimate_rescue_fraction")
        elif metric.key == "burst_damage":
            place("burst_damage_fraction")
        elif metric.key.endswith("aura_eligible_steps"):
            place(metric.key.removesuffix("eligible_steps") + "coverage")

    # These measures do not have a source-to-recipient matrix.
    groups = (
        ("deaths", "death_fraction", "dead_steps", "dead_step_fraction"),
        ("damage_received", "damage_received_fraction"),
        ("healing_received", "healing_received_fraction"),
        ("priest_healing_received", "priest_healing_received_fraction"),
        (
            "effective_priest_healing_received",
            "effective_priest_healing_received_fraction",
        ),
        (
            "excess_healing",
            "excess_healing_allocation_fraction",
            "excess_healing_contribution_fraction",
            "excess_healing_fraction",
            "excess_healing_received",
            "excess_priest_healing_received_fraction",
        ),
        ("regenerated_healing", "regeneration_healing_received_fraction"),
        *(
            (
                (f"{aura}_aura_covered_steps"),
                (f"{aura}_aura_eligible_steps"),
                (f"{aura}_aura_coverage"),
                f"{aura}_aura_covered_steps_allocation_fraction",
                f"{aura}_aura_covered_steps_contribution_fraction",
            )
            for aura in ("mage", "warrior")
        ),
        ("single_contributor_kills", "single_contributor_kill_fraction"),
        ("multi_contributor_kills", "multi_contributor_kill_fraction"),
        ("focus_fire_steps", "focus_fire_concentration"),
        (
            ("actions_submitted"),
            ("actions_accepted"),
            ("action_acceptance_rate"),
            ("actions_rejected"),
            ("action_domain_rejections"),
            ("action_movement_rejections"),
            ("action_combat_rejections"),
        ),
        (
            ("trap_intervals"),
            ("trap_breaks"),
            ("trap_break_rate"),
            ("trap_mean_remaining_steps_at_break"),
        ),
        (
            ("respawn_waves"),
            ("mean_agents_per_respawn_wave"),
            ("mean_observed_respawn_wait_steps"),
        ),
        ("rescue_opportunities", "rescues", "rescue_rate"),
        (
            ("freedom_protected_steps"),
            ("freedom_eligible_steps"),
            ("freedom_protection_fraction"),
        ),
        ("ally_distance_observations", "ally_distance_mean"),
    )
    # Keep companion measures together without sorting their wording.
    ranks = {stem: (rank, 0) for stem, rank in order.items()}
    for group in groups:
        anchor = min((ranks[s][0] for s in group if s in ranks), default=len(ranks))
        for offset, stem in enumerate(group):
            ranks[stem] = (anchor, offset)
    for status in STATUS_NAMES:
        place(status + "_active_steps")
    for column in columns:
        place(column.stem)
    for stem, rank in order.items():
        ranks.setdefault(stem, (rank, 0))
    return ranks


def _build_columns() -> tuple[MetricColumn, ...]:
    """Build the complete immutable scalar schema in its declared order.

    Expand fixed scopes and pair relations, retain the priority prefix, and reuse
    shared totals deliberately. Duplicate names fail unless a caller explicitly
    requests reuse. This host construction changes no numerical metric behavior.
    """
    columns: list[MetricColumn] = []
    names: set[str] = set()

    def add(
        name: str,
        label: str,
        family: str,
        unit: str,
        description: str,
        *,
        direction: MetricDirection,
        scopes: tuple[MetricScope, ...] = ("team", "agent"),
        pairs: tuple[tuple[int, int], ...] = (),
        denominator: str | None = None,
        numerator: str | None = None,
        required_class_id: int | None = None,
        status_channel: int | None = None,
        priority: bool = False,
        missing: str = "",
        requires_ultimate_target: bool = False,
        requires_basic_target: bool = False,
        subject_role: str | None = None,
        recipient_role: str | None = None,
        reuse: bool = False,
        team_description: str | None = None,
        pair_description: str | None = None,
        team_denominator: str | None = None,
    ) -> None:
        """Expand one scalar definition across its declared subjects.

        Required fields supply the stem, label, family, unit and meaning. Optional
        scope/class/target/ratio fields define applicability and readable missing-value
        rules; team/pair wording overrides only those roles. reuse skips an existing
        name, otherwise duplicates raise ValueError. Append only to this local builder.
        """
        if name.startswith("solo_kill"):
            description += " " + _KILL_HELP_TICK_RULE
        if "effective" in name and "healing" in name:
            description += " " + _EFFECTIVE_HEALING_RULE
            if team_description is not None:
                team_description += " " + _EFFECTIVE_HEALING_RULE
            if pair_description is not None:
                pair_description += " " + _EFFECTIVE_HEALING_RULE
        for scope in scopes:
            scope_subject = "this team" if scope == "team" else "this agent"
            scope_denominator = (
                (
                    team_denominator
                    if scope == "team" and team_denominator
                    else denominator
                ).format(subject=scope_subject)
                if denominator is not None
                else None
            )
            subjects_by_scope = (
                ((),)
                if scope == "episode"
                else _TEAMS
                if scope == "team"
                else _AGENTS
                if scope == "agent"
                else pairs
            )
            unavailable = "This measurement was not recorded"
            if scope == "team":
                unavailable += "; nobody is playing on this team"
            elif scope == "agent":
                unavailable += "; this agent was not included in this game"
            elif scope == "source_recipient":
                unavailable += "; either agent was not included in this game"
            elif scope == "team_recipient":
                unavailable += (
                    "; nobody is playing on this team or this target is not in the game"
                )
            elif scope == "ally_pair":
                unavailable += "; either teammate was not included in this game"
            if required_class_id is not None:
                class_name = (
                    "Neutral",
                    "Mage",
                    "Warrior",
                    "Hunter",
                    "Rogue",
                    "Priest",
                )[required_class_id]
                unavailable += (
                    f"; this team has no {class_name} in the game"
                    if scope in ("team", "team_recipient")
                    else f"; this agent is not a {class_name}"
                )
            if scope_denominator is not None:
                unavailable += (
                    f"; the number to divide by ({scope_denominator}) is zero"
                )
            if missing:
                unavailable += f"; {missing}"
            if requires_ultimate_target:
                unavailable += (
                    (
                        "; this target is the wrong kind of target for the relevant "
                        "Ultimates on this team"
                    )
                    if scope == "team_recipient"
                    else (
                        "; this target is the wrong kind of target for this agent's "
                        "Ultimate"
                    )
                )
            if requires_basic_target:
                unavailable += (
                    (
                        "; this target is the wrong kind of target for the "
                        "relevant Basic abilities on this team"
                    )
                    if scope == "team_recipient"
                    else (
                        "; this target is the wrong kind of target for this "
                        "agent's Basic ability"
                    )
                )
            for subjects in subjects_by_scope:
                column_name = _prefix(scope, subjects) + name
                if column_name in names:
                    if reuse:
                        continue
                    raise ValueError(f"duplicate scalar column: {column_name}")
                names.add(column_name)
                columns.append(
                    MetricColumn(
                        name=column_name,
                        label=_team_words(
                            label.format(subject=scope_subject.title()), scope, subjects
                        ),
                        family=family,
                        unit=unit,
                        scope=scope,
                        subjects=subjects,
                        description=_team_words(
                            (
                                team_description
                                if scope == "team" and team_description is not None
                                else pair_description
                                if scope == "ally_pair" and pair_description is not None
                                else description
                            ).format(subject=scope_subject),
                            scope,
                            subjects,
                        ),
                        missing_when=_team_words(
                            unavailable.replace("; ", "; or ") + ".", scope, subjects
                        ),
                        direction=direction,
                        required_class_id=required_class_id,
                        status_channel=status_channel,
                        priority=priority,
                        requires_ultimate_target=requires_ultimate_target,
                        requires_basic_target=requires_basic_target,
                        stem=name,
                        subject_role=subject_role or _subject_role(family, name, scope),
                        recipient_role=recipient_role
                        or (
                            "recipient"
                            if scope in ("source_recipient", "team_recipient")
                            else None
                        ),
                        denominator=_team_words(scope_denominator, scope, subjects)
                        if scope_denominator is not None
                        else None,
                        numerator=_team_words(
                            (
                                numerator or _numerator(name, scope, status_channel)
                            ).format(subject=scope_subject),
                            scope,
                            subjects,
                        )
                        if scope_denominator is not None
                        else None,
                    )
                )

    # Priority order is also the numerical wrapper's fixed result-vector order.
    add(
        "episode_length",
        "Episode Length",
        "priority",
        "steps",
        (
            "How many ticks have been played so far. Calls made after the game "
            "ends do not add time."
        ),
        scopes=("episode",),
        priority=True,
        direction="descriptive",
    )
    for team in (1, 2):
        for outcome in ("win", "draw", "loss"):
            columns.append(
                MetricColumn(
                    name=_prefix("team", (team,)) + outcome,
                    label=outcome.title(),
                    family="priority",
                    unit="indicator",
                    scope="team",
                    subjects=(team,),
                    description=(
                        f"1 if Team {'A' if team == 1 else 'B'} "
                        + {"win": "won", "draw": "drew", "loss": "lost"}[outcome]
                        + "; otherwise 0. Shown only after the game ends."
                    ),
                    missing_when=(
                        "This measurement was not recorded; or the game "
                        "has not ended yet."
                    ),
                    direction=cast(
                        MetricDirection,
                        {"win": "higher", "draw": "descriptive", "loss": "lower"}[
                            outcome
                        ],
                    ),
                    priority=True,
                    stem=outcome,
                    subject_role="team",
                )
            )
            names.add(_prefix("team", (team,)) + outcome)
    add(
        "return",
        "Return",
        "priority",
        "reward",
        (
            "The rewards this agent has earned so far, added together. "
            "Teammates get the same team reward each tick."
        ),
        team_description="The team's rewards so far, added together across ticks. "
        "The team reward counts once each tick, not once for every teammate.",
        scopes=("team",),
        direction="higher",
        priority=True,
    )
    add(
        "score",
        "Score",
        "priority",
        "score",
        (
            "How many points this team has at the selected tick. Includes "
            "points already set when a scenario started."
        ),
        scopes=("team",),
        priority=True,
        direction="higher",
    )
    add(
        "score_difference",
        "Score Difference",
        "priority",
        "score",
        (
            "Team A's points minus Team B's points. For example, 12 minus 8 "
            "gives 4. A positive number means Team A is ahead; a negative "
            "number means Team B is ahead."
        ),
        scopes=("episode",),
        direction="higher",
        priority=True,
    )
    for name, label, description in (
        (
            "kills",
            "Total Kills",
            (
                "How many enemies this team has killed so far. Does not count "
                "points already set when a scenario started."
            ),
        ),
        (
            "deaths",
            "Total Deaths",
            (
                "How many times agents on this team have died, added together. "
                "Dying again after a respawn counts again."
            ),
        ),
    ):
        add(
            name,
            label,
            "priority",
            "count",
            description,
            scopes=("team",),
            priority=True,
            direction=cast(
                MetricDirection, {"kills": "higher", "deaths": "lower"}[name]
            ),
        )

    for ability in ("basic", "ultimate"):
        add(
            f"{ability}_activations",
            f"{ability.title()} Ability Activations",
            "abilities",
            "count",
            f"How many times this agent used its {ability.title()} ability. "
            "Using it again counts once even if its effect is already there. "
            "A use the game does not allow does not count.",
            team_description=f"How many times everyone on this team used their "
            f"{ability.title()} abilities, added together. Using an ability again "
            "counts once even if its effect is already there. A use the game "
            "does not allow does not count.",
            direction="descriptive",
        )

    for class_id, stem in ULTIMATE_TEAM_ABILITY_STEMS.items():
        add(
            stem,
            f"{_ULTIMATE_TOPIC_NAMES[class_id - 1]} Applications",
            "abilities",
            "count",
            f"How many times {ULTIMATE_CLASS_NAMES[class_id - 1].title()}s on "
            f"this team used {_ULTIMATE_NAMES[class_id - 1]}, added together. "
            "Each allowed use counts once. A rejected action does not count.",
            scopes=("team",),
            required_class_id=class_id,
            subject_role="source",
            direction="descriptive",
        )

    add(
        "deaths",
        "Deaths",
        "deaths",
        "count",
        "How many times this agent has died. Dying again after a respawn counts again.",
        scopes=("agent",),
        direction="lower",
    )
    add(
        "death_fraction",
        "Fraction of Team Deaths",
        "deaths",
        "fraction",
        (
            "How much of this team's death count came from this agent. For "
            "example, 2 of the team's 10 deaths gives 2/10 = 0.2 (20%)."
        ),
        scopes=("agent",),
        denominator="all deaths on this team",
        direction="descriptive",
    )
    add(
        "dead_steps",
        "Steps Spent Dead",
        "deaths",
        "agent_steps",
        (
            "How many ticks this agent was dead at the start of the tick. Time "
            "from all of its lives is added together."
        ),
        team_description="Everyone's time spent dead on this team, added together. "
        "Checks who is dead at the start of each tick. Two dead teammates for "
        "one tick count as two ticks of dead time.",
        direction="lower",
    )
    add(
        "dead_step_fraction",
        "Fraction of Team Dead Time",
        "deaths",
        "fraction",
        ("How much of this team's time spent dead came from this agent."),
        scopes=("agent",),
        denominator="everyone's time spent dead on this team, added together",
        direction="descriptive",
    )

    for name, label, unit, description, denominator in (
        (
            "kill_contributions",
            "Kill Contributions",
            "count",
            (
                "How many kills this agent helped with. It must deal damage on the "
                "tick the enemy dies, or give an attacker some useful healing on "
                "that tick. Each agent counts once per enemy death. "
                "Healing another Priest who only provided healing does not count."
            ),
            None,
        ),
        (
            "kill_contributions_per_death",
            "Kill Contributions per Death",
            "ratio",
            (
                "How many kills this agent helped with for each time it died. If "
                "it has not died, this ratio is left blank."
            ),
            "how many times this agent died",
        ),
        (
            "kill_participation",
            "Kill Participation",
            "fraction",
            (
                "What share of this team's kills this agent helped with. Several "
                "teammates can help with the same kill, so their shares can add "
                "up to more than 1 (100%)."
            ),
            "all kills by this team",
        ),
    ):
        add(
            name,
            label,
            "kill_contributions",
            unit,
            description,
            scopes=("agent",),
            denominator=denominator,
            direction="descriptive" if name == "kill_participation" else "higher",
        )

    for effect in ("damage", "healing"):
        description = {
            "damage": (
                "How much damage {subject} dealt after damage boosts and defenses. "
                "Includes damage beyond the enemy's remaining health. More damage "
                "does not always mean more kills."
            ),
            "healing": (
                "How much Priest healing {subject} provided after effects such as "
                "Poison reduced it. Includes healing that did not fit below maximum"
                " health. Does not include automatic regeneration."
            ),
        }[effect]
        add(
            f"{effect}_done",
            f"{effect.title()} Done",
            f"{effect}_done",
            "health",
            description,
            direction=cast(
                MetricDirection, {"damage": "higher", "healing": "descriptive"}[effect]
            ),
        )
        add(
            f"{effect}_done_fraction",
            f"Fraction of Team {effect.title()}",
            f"{effect}_done",
            "fraction",
            f"How much of this team's {effect} came from this agent.",
            scopes=("agent",),
            denominator=f"all {effect} this team provided",
            direction="descriptive",
        )
        recipients = tuple(
            (source, recipient)
            for source in range(10)
            for recipient in range(10)
            if (source // 5 == recipient // 5) == (effect == "healing")
        )
        add(
            f"{effect}_done",
            f"{effect.title()} Done to Recipient",
            f"recipient_{effect}",
            "health",
            (
                "How much damage this agent dealt to this enemy, after damage "
                "boosts and defenses. "
                if effect == "damage"
                else "How much Priest healing this agent gave this ally, after "
                "effects such as Poison reduced it. "
            )
            + description.split(". ", 1)[1],
            scopes=("source_recipient",),
            pairs=recipients,
            direction="higher" if effect == "damage" else "descriptive",
        )
        add(
            f"{effect}_done_fraction",
            f"Fraction of Team {effect.title()} to Recipient",
            f"recipient_{effect}",
            "fraction",
            f"How much of the team's {effect} to this target came from this "
            "agent. The team total includes this agent.",
            scopes=("source_recipient",),
            pairs=recipients,
            denominator=f"all {effect} this team provided to this target",
            direction="descriptive",
        )

    add(
        "damage_received",
        "Damage Received",
        "damage_received",
        "health",
        (
            "How much damage {subject} took after damage boosts and defenses. "
            "Includes damage beyond the health they had left."
        ),
        direction="lower",
    )
    add(
        "damage_received_fraction",
        "Fraction of Team Damage Received",
        "damage_received",
        "fraction",
        (
            "How much of this team's incoming damage hit this agent. Taking a "
            "larger share is not always good or bad."
        ),
        scopes=("agent",),
        denominator="all damage this team took",
        direction="descriptive",
    )
    for name, label, description in (
        (
            "priest_healing_received",
            "Priest Healing Received",
            (
                "How much Priest healing {subject} received after effects such as "
                "Poison reduced it. Includes healing that did not fit below maximum"
                " health. Does not include regeneration."
            ),
        ),
        (
            "regenerated_healing",
            "Regenerated Healing",
            (
                "How much health {subject} actually got back from automatic "
                "regeneration, after that tick's damage and Priest healing."
            ),
        ),
        (
            "healing_received",
            "Total Healing Received",
            (
                "All Priest healing {subject} received, including excess healing, "
                "plus health actually restored by regeneration."
            ),
        ),
    ):
        add(
            name,
            label,
            "healing_received",
            "health",
            description,
            direction="descriptive",
        )
    for source in ("priest", "regeneration"):
        add(
            f"{source}_healing_received_fraction",
            f"{source.title()} Fraction of Healing Received",
            "healing_received",
            "fraction",
            f"How much of {{subject}}'s total healing came from "
            f"{'Priests' if source == 'priest' else 'regeneration'}. "
            "Priest healing includes excess; regeneration counts only health restored.",
            denominator=(
                "Priest healing {subject} received, including excess, plus health "
                "restored by regeneration"
            ),
            direction="descriptive",
        )
    add(
        "healing_received_fraction",
        "Fraction of Team Healing Received",
        "healing_received",
        "fraction",
        (
            "How much of this team's total healing this agent received. "
            "Priest healing includes excess; regeneration counts only health "
            "restored."
        ),
        scopes=("agent",),
        denominator=(
            "the whole team's Priest healing, including excess, plus health "
            "restored by regeneration"
        ),
        direction="descriptive",
    )

    add(
        "excess_healing",
        "Excess Healing",
        "excess_healing",
        "health",
        (
            "How much of the healing {subject} provided could not fit below "
            "allies' maximum health after that tick's damage and healing. "
            "Priests healing the same ally share the excess in proportion to how"
            " much each healed. Does not include regeneration."
        ),
        required_class_id=5,
        direction="descriptive",
    )
    add(
        "excess_healing_received",
        "Excess Healing Received",
        "excess_healing",
        "health",
        (
            "How much Priest healing this agent received that could not fit "
            "below its maximum health after that tick's damage and healing. "
            "Does not include regeneration."
        ),
        scopes=("agent",),
        direction="descriptive",
    )

    for channel, status in enumerate(STATUS_NAMES[:7]):
        for effect in ("damage", "healing"):
            add(
                f"{effect}_to_{status}_recipient",
                f"{effect.title()} to "
                f"{'Enemies' if effect == 'damage' else 'Allies'} With This Effect",
                f"controlled_{effect}",
                "health",
                (
                    (
                        "How much damage {subject} dealt to enemies who "
                        "already had this status at the start of the tick. "
                        "Includes damage beyond their remaining health. "
                        "Damage that breaks an existing Trap still counts. "
                        "One attack can count under several statuses."
                    )
                    if effect == "damage"
                    else (
                        "How much Priest healing {subject} provided to allies"
                        " who already had this harmful status at the start of"
                        " the tick. Includes excess healing, but not "
                        "regeneration. One heal can count under several "
                        "statuses."
                    )
                ),
                required_class_id=5 if effect == "healing" else None,
                status_channel=channel,
                direction="higher"
                if effect == "damage" and status != "hunter_trap"
                else "descriptive",
            )
        add(
            f"kills_of_{status}_recipient",
            "Kills of Enemies With This Effect",
            "controlled_kills",
            "count",
            (
                "How many enemies this team killed while they already had this "
                "status at the start of the tick. Each death counts once for this "
                "status, even if several teammates helped."
            ),
            scopes=("team",),
            status_channel=channel,
            direction="higher",
        )
        add(
            f"kill_contributions_to_{status}_recipient",
            "Kill Contributions Against Enemies With This Effect",
            "controlled_kills",
            "count",
            (
                "How many kills this agent helped with when the enemy already had "
                "this status at the start of the tick. Healing an attacker on the "
                "kill tick also counts if the heal was at least partly useful."
            ),
            scopes=("agent",),
            status_channel=channel,
            direction="higher",
        )
        add(
            f"kill_participation_in_{status}_recipient",
            "Share of Team Kills Against Enemies With This Effect",
            "controlled_kills",
            "fraction",
            (
                "What share of this team's kills against enemies with this status "
                "involved this agent. The enemy must already have the status at "
                "the start of the tick. Several agents can help with one kill, so "
                "their shares may add up to more than 1 (100%)."
            ),
            scopes=("agent",),
            denominator=(
                "all kills by this team against enemies who already had this status"
                " at the start of the tick"
            ),
            status_channel=channel,
            direction="descriptive",
        )

    for name, label, unit, description, denominator in (
        (
            "single_contributor_kills",
            "Single-Contributor Kills",
            "count",
            (
                "How many enemies this team killed with exactly one agent helping. "
                "A Priest's healing on the kill tick counts as help if it reaches "
                "an attacker and is at least partly useful."
            ),
            None,
        ),
        (
            "multi_contributor_kills",
            "Multiple-Contributor Kills",
            "count",
            (
                "How many enemies this team killed with at least two agents "
                "helping. A Priest's healing on the kill tick counts as help if it "
                "reaches an attacker and is at least partly useful."
            ),
            None,
        ),
        (
            "single_contributor_kill_fraction",
            "Single-Contributor Kill Fraction",
            "fraction",
            (
                "What share of this team's kills had exactly one helper. Solo kills "
                "are not always better or worse than shared kills."
            ),
            "all kills by this team",
        ),
        (
            "multi_contributor_kill_fraction",
            "Multiple-Contributor Kill Fraction",
            "fraction",
            (
                "What share of this team's kills had two or more helpers. Shared "
                "kills are not always better or worse than solo kills."
            ),
            "all kills by this team",
        ),
        (
            "focus_fire_concentration",
            "Focus Fire Concentration",
            "fraction",
            (
                "How often damaging teammates on this team attack the same enemy. "
                "On each tick with at least two attackers, measure the share "
                "attacking the most-shared enemy. Average those tick results. Two "
                "attackers hitting different enemies gives 0.5 (50%); both "
                "hitting one enemy gives 1 (100%). Each tick counts equally, "
                "regardless of damage dealt."
            ),
            "ticks when at least two agents on this team dealt damage",
        ),
        (
            "focus_fire_steps",
            "Multi-Attacker Ticks",
            "steps",
            (
                "How many ticks at least two agents on this team dealt damage. "
                "Counts what they actually did, not chances they might have had "
                "to attack together."
            ),
            None,
        ),
    ):
        if name in (
            "single_contributor_kills",
            "multi_contributor_kills",
            "single_contributor_kill_fraction",
            "multi_contributor_kill_fraction",
        ):
            description += " " + _KILL_HELP_TICK_RULE
        add(
            name,
            label,
            "coordination",
            unit,
            description,
            scopes=("team",),
            denominator=denominator,
            direction=cast(
                MetricDirection,
                {
                    "single_contributor_kills": "higher",
                    "multi_contributor_kills": "higher",
                    "single_contributor_kill_fraction": "descriptive",
                    "multi_contributor_kill_fraction": "descriptive",
                    "focus_fire_concentration": "descriptive",
                    "focus_fire_steps": "descriptive",
                }[name],
            ),
        )

    for name, label, description in (
        (
            "actions_submitted",
            "Actions Submitted",
            (
                "How many actions this agent sent to the game. Counts one per tick,"
                " including do-nothing actions while dead."
            ),
        ),
        (
            "actions_accepted",
            "Actions Fully Accepted",
            (
                "How many of this agent's actions had both movement and ability "
                "choices allowed. A legal do-nothing action counts too."
            ),
        ),
        (
            "actions_rejected",
            "Actions Rejected",
            (
                "How many of this agent's actions had at least one choice rejected."
                " An action counts once here even if several choices were rejected."
            ),
        ),
        (
            "action_domain_rejections",
            "Out-of-Domain Rejections",
            (
                "How many of this agent's actions contained a choice the game does "
                "not recognize. The whole action is rejected."
            ),
        ),
        (
            "action_movement_rejections",
            "Movement Rejections",
            (
                "How many of this agent's actions chose a known movement that was "
                "not allowed at that time. The same action may also have a rejected"
                " ability choice."
            ),
        ),
        (
            "action_combat_rejections",
            "Combat Rejections",
            (
                "How many of this agent's actions chose a known target or Ultimate "
                "option that was not allowed at that time. The same action may also"
                " have rejected movement."
            ),
        ),
    ):
        add(
            name,
            label,
            "action_acceptance",
            "count",
            description,
            team_description=description.replace("this agent's", "this team's")
            .replace("this agent sent", "everyone on this team sent")
            .replace("Counts one per tick", "Counts one per agent per tick"),
            direction=cast(
                MetricDirection,
                {
                    "actions_submitted": "descriptive",
                    "actions_accepted": "higher",
                    "actions_rejected": "lower",
                    "action_domain_rejections": "lower",
                    "action_movement_rejections": "lower",
                    "action_combat_rejections": "lower",
                }[name],
            ),
        )
    add(
        "action_acceptance_rate",
        "Action Acceptance Rate",
        "action_acceptance",
        "fraction",
        (
            "How often every choice in an action {subject} sent was allowed. "
            "Legal do-nothing actions count as allowed."
        ),
        denominator="all actions {subject} sent",
        team_denominator="all actions sent by agents on this team",
        direction="higher",
    )

    for channel, status in enumerate(STATUS_NAMES):
        add(
            f"{status}_applications",
            STATUS_LABELS[channel] + " Applications",
            "status_applications",
            "count",
            "How many times this agent used the ability that gives this status. "
            "Using it again counts once even if the status is already there. "
            "One use can give several statuses. This does not count how many "
            "separate periods the status lasts.",
            team_description="How many times agents on this team used the ability "
            "that gives this status, added together. Using it again counts once "
            "even if the status is already there. One use can give several statuses.",
            required_class_id=STATUS_CLASS_IDS[channel],
            status_channel=channel,
            direction="descriptive",
        )
        add(
            f"{status}_active_steps",
            "Time With Status On {subject}",
            "status_active_steps",
            "agent_steps",
            (
                "How many ticks this agent was alive and already had this status at"
                " the start of the tick. This measures time for the agent with the "
                "status, not the agent who applied it."
            )
            + (
                " A slow can be present without reducing movement, for example "
                "when Freedom protects the agent."
                if channel < 3
                else ""
            ),
            team_description="Everyone's time with this status on this team, "
            "added together. An agent must be alive and already have the status "
            "at the start of the tick. Two affected teammates for one tick count "
            "as two. This describes agents with the status, not whoever gave it."
            + (
                " A slow can be present without reducing movement, for example "
                "when Freedom protects the agent."
                if channel < 3
                else ""
            ),
            required_class_id=1 if channel == 7 else None,
            status_channel=channel,
            direction=cast(
                MetricDirection,
                (
                    "lower",
                    "lower",
                    "lower",
                    "lower",
                    "lower",
                    "lower",
                    "lower",
                    "descriptive",
                    "descriptive",
                )[channel],
            ),
        )

    for name, label, unit, description, denominator in (
        (
            "trap_intervals",
            "Times Enemies Were Trapped",
            "count",
            (
                "How many separate times enemies of this team were trapped. "
                "Includes traps already there at the start and traps still active "
                "now. Two Hunters trapping one enemy on the same tick count as "
                "one period and two activations. A later Trap cast deals damage "
                "before applying its Trap: it breaks the old Trap or follows its "
                "expiry, then starts a new period."
            ),
            None,
        ),
        (
            "trap_breaks",
            "Traps Broken by Damage",
            "count",
            (
                "How many Traps on enemies this team broke with damage. Each break "
                "counts once even if several teammates helped. A Trap that ran out "
                "on its own does not count."
            ),
            None,
        ),
        (
            "trap_break_rate",
            "Trap Break Rate",
            "fraction",
            (
                "What share of Traps on enemies ended because this team broke "
                "the Trap with damage. Includes traps already active at the start "
                "and traps still open. Breaking a Trap may free an enemy or help "
                "kill them."
            ),
            (
                "all Trap periods on enemies seen so far, including those already "
                "active at the start or still open"
            ),
        ),
        (
            "trap_mean_remaining_steps_at_break",
            "Mean Trap Time Left at Break",
            "steps",
            (
                "How many ticks traps had left when this team broke them with "
                "damage, averaged across breaks. Uses the time left after the timer"
                " went down for that tick. Traps that ran out on their own do not "
                "count."
            ),
            "Traps on enemies broken by damage",
        ),
    ):
        add(
            name,
            label,
            "trap_breaks",
            unit,
            description,
            scopes=("team",),
            denominator=denominator,
            direction="descriptive",
        )
    for stem, label, description, numerator, denominator in (
        (
            "trap_intervals",
            "Times This Agent Was Trapped",
            "How many separate times this agent was trapped. Includes a Trap "
            "already active when recording began and a Trap still active now. "
            "Two Hunters trapping it on the same tick start one period. A later "
            "Trap cast breaks the old Trap or follows expiry, then starts another.",
            None,
            None,
        ),
        (
            "trap_break_rate",
            "Share of This Agent's Traps Broken by Damage",
            "What share of this agent's Trap periods ended because damage broke "
            "the Trap. Includes periods already active at the start or still open. "
            "Example: one break out of two periods gives 0.5, or 50%.",
            "times damage broke a Trap on this agent",
            "all separate Trap periods on this agent seen so far",
        ),
        (
            "trap_mean_remaining_steps_at_break",
            "Mean Trap Time Left When Broken",
            "How many ticks this agent's Trap had left when damage broke it, "
            "averaged across breaks. Uses the time after that tick's timer "
            "decrease. Natural expiry does not count as a break.",
            "ticks left on each Trap broken on this agent, added together, "
            "after that tick's timer decrease",
            "times damage broke a Trap on this agent",
        ),
    ):
        add(
            stem,
            label,
            "trap_breaks",
            "count"
            if denominator is None
            else "fraction"
            if stem == "trap_break_rate"
            else "steps",
            description,
            scopes=("agent",),
            direction="descriptive",
            numerator=numerator,
            denominator=denominator,
            subject_role="recipient",
            recipient_role="trapped_enemy",
        )
    add(
        "trap_break_contributions",
        "Trap Break Contributions",
        "trap_breaks",
        "count",
        (
            "How many enemy traps this agent helped break with damage. Several "
            "agents can help break the same Trap."
        ),
        scopes=("agent",),
        direction="descriptive",
    )
    add(
        "trap_break_participation",
        "Trap Break Participation",
        "trap_breaks",
        "fraction",
        (
            "What share of this team's Trap breaks this agent helped with. "
            "Several teammates can share one break."
        ),
        scopes=("agent",),
        denominator="all enemy traps this team broke with damage",
        direction="descriptive",
    )

    add(
        "respawn_waves",
        "Respawn Waves",
        "respawn",
        "count",
        "How many scheduled chances this team has had to respawn. "
        "Counts a wave even when no agent returns.",
        scopes=("team",),
        direction="descriptive",
    )
    add(
        "mean_agents_per_respawn_wave",
        "Mean Agents per Respawn Wave",
        "respawn",
        "agents",
        (
            "The average number of agents brought back by each of this team's "
            "respawn waves. Includes scheduled waves when no agent returns."
        ),
        scopes=("team",),
        denominator="all scheduled respawn waves this team had, including empty waves",
        direction="descriptive",
    )

    for name, label, unit, description, denominator in (
        (
            "burst_damage",
            "Damage While Burst Is Active",
            "health",
            (
                "How much damage {subject} dealt while Burst was already active at "
                "the start of the tick. Includes damage beyond the enemy's "
                "remaining health. This is all damage during Burst, not just the "
                "extra damage it adds. Turning on Burst itself deals no damage."
            ),
            None,
        ),
        (
            "burst_damage_fraction",
            "Burst Fraction of Mage Damage",
            "fraction",
            (
                "How much of this Mage's damage came while Burst was already "
                "active. This shows how much it relied on Burst, not how well it "
                "played."
            ),
            "all damage this Mage dealt",
        ),
        (
            "burst_damage_contributing_to_kill",
            "Burst Damage on Enemy Death Ticks",
            "health",
            (
                "How much damage {subject} dealt with Burst active to enemies on "
                "the tick those enemies died. Includes damage beyond their "
                "remaining health and all damage boosts. It is not just the extra "
                "damage Burst adds, or proof that Burst was needed for the kill."
            ),
            None,
        ),
        (
            "burst_kill_contributions",
            "Kill Contributions During Burst",
            "count",
            (
                "How many kills this Mage helped with by dealing damage while Burst"
                " was already active at the start of the tick. The Mage must deal "
                "damage on the tick that enemy dies."
            ),
            None,
        ),
    ):
        add(
            name,
            label,
            "burst",
            unit,
            description,
            team_description=(
                "How many kills this team's Mages helped with by dealing damage "
                "while Burst was already active at the start of the tick. Adds "
                "each Mage's count, so two Mages helping with one kill count as two. "
                "Only damage on the tick the enemy dies counts as help."
                if name == "burst_kill_contributions"
                else (
                    "How much of this team's Mage damage came while Burst was already "
                    "active. This shows how much the Mages relied on Burst, not how "
                    "well they played."
                )
                if name == "burst_damage_fraction"
                else description.replace("{subject}", "this team's Mages")
            ),
            denominator=denominator,
            team_denominator="all damage this team's Mages dealt"
            if name == "burst_damage_fraction"
            else None,
            required_class_id=1,
            direction=cast(
                MetricDirection,
                {
                    "burst_damage": "higher",
                    "burst_damage_fraction": "descriptive",
                    "burst_damage_contributing_to_kill": "higher",
                    "burst_kill_contributions": "higher",
                }[name],
            ),
        )

    for aura, class_id in (("mage", 1), ("warrior", 2)):
        for suffix, label, unit, description, denominator in (
            (
                "covered_steps",
                f"{aura.title()} Aura Coverage Across Allies",
                "agent_steps",
                "How many teammates this agent's aura covered at the start of "
                "each tick, added together across ticks. Both agents must be "
                "alive and have no spawn shield. The aura giver can count itself.",
                None,
            ),
            (
                "eligible_steps",
                f"Ticks Allies Could Receive {aura.title()} Aura",
                "agent_steps",
                "How many teammates this agent's aura could cover, added "
                "together across ticks. Both agents must be alive and have no "
                "spawn shield at the start of the tick. They do not have to be "
                "in aura range. The aura giver can count itself.",
                None,
            ),
            (
                "coverage",
                f"{aura.title()} Aura Coverage",
                "fraction",
                (
                    "How often this agent's aura covered its allies. The aura giver "
                    "can cover itself. More coverage alone does not prove better "
                    "positioning."
                ),
                (
                    "ticks each ally could receive this agent's aura, added together; "
                    "count each ally once per tick while it and the giver are alive "
                    "without spawn shields, including allies out of range and the "
                    "giver itself"
                ),
            ),
        ):
            add(
                f"{aura}_aura_{suffix}",
                label,
                "aura_coverage",
                unit,
                description,
                team_description={
                    ("covered_steps"): (
                        "How many agents on this team at least one "
                        f"{aura.title()}"
                        " aura covered at the start of each tick, added together. "
                        "Count each teammate once per tick even with several aura "
                        "givers. Both "
                        "the covered agent and an aura giver must be alive and have no "
                        "spawn shield. A giver can cover itself."
                    ),
                    "eligible_steps": (
                        "How many agents on this team could receive a "
                        f"{aura.title()}"
                        " aura at the start of each tick, added together. Count "
                        "each teammate once per tick if it and at least one aura "
                        "giver are alive and have no spawn shield. Distance does "
                        "not matter. An aura giver can count itself."
                    ),
                    "coverage": (
                        f"How often this team's agents had a {aura.title()} aura. "
                        "Each agent counts once per tick, even with several auras. "
                        "Both the teammate and at least one aura giver must be "
                        "alive and have no spawn shield. "
                        "Givers can count "
                        "themselves."
                    ),
                }[suffix],
                denominator=denominator,
                team_denominator=(
                    f"ticks agents on this team could receive a {aura.title()} aura, "
                    "added together; each agent counts once per tick while it and "
                    "at least one aura giver are alive without spawn shields, "
                    "including agents out of range"
                )
                if denominator
                else None,
                required_class_id=class_id,
                direction="descriptive",
            )
    add(
        "damage_from_mage_aura",
        "Damage Resulting from Mage Aura",
        "aura_benefits",
        "health",
        (
            "Extra damage this team dealt because of Mage auras. Compares the "
            "same attacks with the auras turned off, keeping Burst and enemy "
            "defenses the same. Measures overlapping auras together."
        ),
        scopes=("team",),
        required_class_id=1,
        direction="higher",
    )
    add(
        "damage_prevented_by_warrior_aura",
        "Damage Prevented by Warrior Aura",
        "aura_benefits",
        "health",
        (
            "Damage this team avoided because of Warrior auras. Compares the "
            "same attacks with those auras turned off. Measures overlapping "
            "auras together."
        ),
        scopes=("team",),
        required_class_id=2,
        direction="higher",
    )
    add(
        "healing_prevented_by_poison",
        "Priest Healing Prevented On {subject}",
        "poison",
        "health",
        (
            "How much Rogue Poison reduced the Priest healing sent to {subject}. "
            "Poison must already be active at the start of the tick. Measures "
            "the reduction before the maximum-health limit, including healing "
            "that would have been excess. Example: Poison reduces a heal from "
            "8 to 4, so 4 counts, even if the ally was already at full health. "
            "Does not include regeneration or guess which Rogue applied Poison."
        ),
        direction="lower",
    )

    for name, label, unit, description, denominator in (
        (
            "rescue_opportunities",
            "Lethal Damage Rescue Opportunities",
            "count",
            (
                "How many times incoming damage would kill an ally and the team had"
                " enough Priest healing ready and allowed to save them. Several "
                "Priests may be needed. Each threatened ally counts once per tick. "
                "Each ally is checked separately. This does not mean the Priests "
                "could save all threatened allies at once."
            ),
            None,
        ),
        (
            "rescues",
            "Lethal Damage Rescues",
            "count",
            (
                "How many times Priest healing on this team kept an ally alive "
                "when that tick's damage would otherwise kill them. Each saved "
                "ally counts once per tick."
            ),
            None,
        ),
        (
            "rescue_rate",
            "Lethal Damage Rescue Rate",
            "fraction",
            (
                "How often this team saved an ally when its available Priest "
                "healing could have saved that ally from death."
            ),
            (
                "times this team had enough healing ready and allowed to save an "
                "ally from death"
            ),
        ),
    ):
        add(
            name,
            label,
            "priest_rescue",
            unit,
            description,
            scopes=("team",),
            denominator=denominator,
            required_class_id=5,
            direction=cast(
                MetricDirection,
                {
                    "rescue_opportunities": "descriptive",
                    "rescues": "higher",
                    "rescue_rate": "higher",
                }[name],
            ),
        )
    add(
        "rescue_opportunities",
        "Chances to Save This Agent",
        "priest_rescue",
        "count",
        "How many times this agent would die from the incoming damage, but its "
        "team had enough Priest healing ready and allowed to save it. Checks "
        "legal range, targets, cooldowns, stuns, shields and healing reductions. "
        "Several Priests may be needed. Each ally is checked separately; the "
        "team may not be able to save every threatened ally at once. Includes "
        "Basic and Ultimate healing. Regeneration does not count.",
        scopes=("agent",),
        direction="descriptive",
        subject_role="recipient",
        recipient_role="saved_ally",
        missing="this team has no Priest in the game",
    )
    add(
        "rescue_contributions",
        "Lethal Damage Rescue Contributions",
        "priest_rescue",
        "count",
        (
            "How many times this Priest helped save an ally from death by "
            "healing them on that tick. At least part of the heal must be "
            "useful. Several Priests can help; one helper may not have had "
            "enough healing to save the ally alone."
        ),
        scopes=("agent",),
        required_class_id=5,
        direction="higher",
    )
    add(
        "rescue_participation",
        "Lethal Damage Rescue Participation",
        "priest_rescue",
        "fraction",
        (
            "What share of this team's rescues this Priest helped with. Several "
            "Priests can help with the same rescue."
        ),
        scopes=("agent",),
        denominator="all rescues by this team, counting each saved ally once per tick",
        required_class_id=5,
        direction="descriptive",
    )
    for name, label, unit, description, denominator in (
        (
            "freedom_protected_steps",
            "Freedom Slow Protection Steps",
            "agent_steps",
            (
                "How many ticks Freedom let this agent move faster despite a slow. "
                "Counts even if the agent stood still. This is not distance "
                "traveled."
            ),
            None,
        ),
        (
            "freedom_eligible_steps",
            "Freedom Eligible Steps",
            "agent_steps",
            (
                "How many ticks this agent had Freedom and was alive, without a "
                "stun or spawn shield. Counts even if it stood still or was not "
                "slowed."
            ),
            None,
        ),
        (
            "freedom_protection_fraction",
            "Freedom Slow Protection Fraction",
            "fraction",
            (
                "How often Freedom weakened a slow on this agent while it was "
                "alive, without a stun or spawn shield. This also depends on how "
                "much enemies slowed the agent."
            ),
            (
                "ticks with Freedom while alive, without a stun or spawn shield, "
                "including ticks without a slow"
            ),
        ),
    ):
        add(
            name,
            label,
            "freedom",
            unit,
            description,
            team_description=description.replace("this agent", "each agent")
            + (
                " Adds everyone's counted ticks on this team."
                if denominator
                else " Adds everyone's ticks on this team."
            ),
            denominator=denominator,
            team_denominator=(
                "ticks each agent on this team had Freedom while alive without "
                "a stun or spawn shield, added together, including ticks without a slow"
            )
            if denominator
            else None,
            direction="descriptive",
        )

    add(
        "ally_distance_mean",
        "Mean Ally Distance",
        "formation",
        "distance",
        (
            "The average distance between living teammates on this team at "
            "the start of each tick, in map distance units. Every pair is "
            "measured once per tick, and every measurement has equal weight."
        ),
        pair_description="The average distance between these two agents while "
        "both are alive, measured at the start of each tick. Uses map distance "
        "units. Spawn shields do not exclude them.",
        scopes=("team", "ally_pair"),
        pairs=_ALLY_PAIRS,
        denominator="distance measurements made while both teammates were alive",
        team_denominator="distance measurements across living ally pairs on this team",
        direction="descriptive",
    )
    add(
        "ally_distance_observations",
        "Ally-Pair Distance Measurements",
        "formation",
        "pair_steps",
        (
            "How many distances were measured between living teammates on "
            "this team at the start of each tick. Each pair counts once. "
            "Three living teammates give three measurements per tick. Spawn "
            "shields do not exclude teammates."
        ),
        pair_description="How many ticks both of these agents were alive at the "
        "start. Their distance is measured once on each of those ticks. Spawn "
        "shields do not exclude them.",
        scopes=("team", "ally_pair"),
        pairs=_ALLY_PAIRS,
        direction="descriptive",
    )

    # Schema 2 appends measurements; the first 1,388 names retain their order.
    for name, label, family, description, denominator, direction in (
        (
            "effective_healing_done",
            "Effective Healing Done",
            "healing_done",
            (
                "How much Priest healing {subject} provided, minus the part that "
                "was more than needed. If several Priests heal the same ally on a "
                "tick, they "
                "share the excess in proportion to how much each healed. Does not "
                "include regeneration."
            ),
            None,
            "higher",
        ),
        (
            "effective_healing_fraction",
            "Effective Fraction of Healing Done",
            "healing_done",
            ("How much of the Priest healing {subject} provided was useful."),
            "all Priest healing {subject} provided, including excess",
            "descriptive",
        ),
        (
            "excess_healing_fraction",
            "Excess Fraction of Healing Done",
            "excess_healing",
            (
                "How much of the Priest healing {subject} provided was more than "
                "needed. A Basic heal can still give the ally Freedom even when "
                "the ally needs none of its healing."
            ),
            "all Priest healing {subject} provided, including excess",
            "descriptive",
        ),
        (
            "effective_priest_healing_received",
            "Effective Priest Healing Received",
            "healing_received",
            (
                "Priest healing {subject} received, minus the healing that could "
                "not fit below maximum health after that tick's damage and healing."
                " Does not include regeneration."
            ),
            None,
            "higher",
        ),
        (
            "effective_priest_healing_received_fraction",
            "Effective Fraction of Priest Healing Received",
            "healing_received",
            ("How much of the Priest healing {subject} received was useful."),
            "all Priest healing {subject} received, including excess",
            "descriptive",
        ),
        (
            "excess_priest_healing_received_fraction",
            "Excess Fraction of Priest Healing Received",
            "healing_received",
            ("How much of the Priest healing {subject} received was more than needed."),
            "all Priest healing {subject} received, including excess",
            "descriptive",
        ),
        (
            "effective_healing_received",
            "Total Effective Healing Received",
            "healing_received",
            (
                "Priest healing {subject} received that was useful, plus health"
                " actually restored by regeneration. Useful Priest healing and "
                "regeneration answer different questions. More regeneration may "
                "mean good disengagement or too much time away from the fight."
            ),
            None,
            "descriptive",
        ),
    ):
        add(
            name,
            label,
            family,
            "fraction" if denominator else "health",
            description,
            denominator=denominator,
            direction=cast(MetricDirection, direction),
        )

    add(
        "solo_kills",
        "Solo Kills",
        "kill_contributions",
        "count",
        (
            "How many kills this agent got without another agent helping. A "
            "Priest who gives the attacker some useful healing on that tick "
            "also counts as a helper, so that kill is not solo."
        ),
        scopes=("agent",),
        direction="higher",
    )
    add(
        "solo_kill_fraction",
        "Solo Fraction of Kill Contributions",
        "kill_contributions",
        "fraction",
        "What share of this agent's kills had no other helper.",
        scopes=("agent",),
        denominator="all kills this agent helped with",
        direction="descriptive",
    )

    add(
        "mean_observed_respawn_wait_steps",
        "Mean Wait Before Respawn",
        "respawn",
        "steps",
        (
            "How many ticks agents on this team waited to respawn, on average. "
            "Each wait counts once. Includes waits still in progress, using only "
            "the time seen so far. For agents already dead at the start, time "
            "before the recording is not counted. A death on the last tick "
            "starts a wait with zero ticks counted so far. "
            "Example: waits of 2 and 4 ticks give an average of 3 ticks."
        ),
        scopes=("team",),
        denominator=(
            "waits for respawn seen for each agent on this team, added together, "
            "including unfinished waits and agents already dead at the start"
        ),
        direction="descriptive",
    )

    all_recipients = tuple(
        (source, recipient) for source in range(10) for recipient in range(10)
    )
    enemy_recipients = tuple(
        pair for pair in all_recipients if pair[0] // 5 != pair[1] // 5
    )
    ally_recipients = tuple(
        pair for pair in all_recipients if pair[0] // 5 == pair[1] // 5
    )
    add(
        "ultimate_applications",
        "Ultimate Ability Activations on Recipient",
        "abilities",
        "count",
        "How many times this agent used its Ultimate on this target. "
        "Using it again counts once even if the effect is already there. A use "
        "the game does not allow does not count. Mages use Burst on themselves; "
        "turning it on does not deal damage.",
        scopes=("source_recipient",),
        pairs=all_recipients,
        direction="descriptive",
        requires_ultimate_target=True,
    )
    add(
        "ultimate_application_fraction",
        "Ultimate Ability Activation Allocation",
        "abilities",
        "fraction",
        (
            "How much of this agent's Ultimate ability use was on this "
            "target. For example, 4 of its 10 activations gives 4/10 = 0.4 "
            "(40%)."
        ),
        scopes=("source_recipient",),
        pairs=all_recipients,
        denominator="all Ultimate ability activations by this agent",
        direction="descriptive",
        requires_ultimate_target=True,
    )
    for effect, recipients, description, direction in (
        (
            "damage",
            enemy_recipients,
            (
                "How much damage {subject} dealt directly with Ultimates, after "
                "damage boosts and defenses. Includes damage beyond the enemy's "
                "remaining health. Turning on Burst deals no damage; attacks made "
                "later with Burst active count under Burst Damage."
            ),
            "higher",
        ),
        (
            "healing",
            ally_recipients,
            (
                "How much healing {subject} provided with Salvation. Includes "
                "healing that did not fit below maximum health. Does not include "
                "regeneration."
            ),
            "descriptive",
        ),
    ):
        add(
            f"ultimate_{effect}_done",
            f"Ultimate {effect.title()} Done",
            f"{effect}_done",
            "health",
            description,
            direction=cast(MetricDirection, direction),
        )
        add(
            f"ultimate_{effect}_done",
            f"Ultimate {effect.title()} Done to Recipient",
            f"recipient_{effect}",
            "health",
            description.replace(
                "{subject} dealt directly with Ultimates",
                "this agent dealt to this enemy directly with Ultimates",
            ).replace(
                "{subject} provided with Salvation",
                "this agent gave this ally with Salvation",
            ),
            scopes=("source_recipient",),
            pairs=recipients,
            direction=cast(MetricDirection, direction),
        )

    add(
        "basic_kill_participation",
        "Basic Kill Participation",
        "kill_contributions",
        "fraction",
        (
            "What share of this team's kills this agent helped with using its "
            "Basic ability. It must damage the enemy, or give an attacker some "
            "useful healing, on the tick the enemy dies. A Mage's Basic "
            "attack during Burst counts here. Several teammates can help with "
            "the same kill."
        ),
        scopes=("agent",),
        denominator="all kills by this team",
        direction="descriptive",
        subject_role="contributor",
    )
    add(
        "basic_kill_fraction",
        "Share of Kills Helped by Basic Abilities",
        "kill_contributions",
        "fraction",
        (
            "What share of this team's kills had help from a Basic ability on "
            "the tick the enemy died. Help means damaging the enemy or giving "
            "an attacker some useful healing. Count each enemy death "
            "once, even if several teammates helped. A kill can have both Basic "
            "and Ultimate help, so those shares can add up to more than 100%."
        ),
        scopes=("team",),
        denominator="all kills by this team",
        direction="descriptive",
        subject_role="team",
    )

    add(
        "basic_rescue_participation",
        "Share of Team Saves Helped by This Priest's Basic Healing",
        "priest_rescue",
        "fraction",
        "What share of this team's saved allies this Priest helped save with "
        "Basic healing on that tick. At least part of its healing must be useful. "
        "Several Priests can help save the same ally.",
        scopes=("agent",),
        numerator="saves this Priest helped with using Basic healing",
        denominator="all unique saves by this team, from Basic and Ultimate healing",
        required_class_id=5,
        direction="descriptive",
        subject_role="contributor",
    )
    add(
        "basic_rescue_fraction",
        "Share of Saves Helped by Basic Healing",
        "priest_rescue",
        "fraction",
        "What share of this team's saves had help from Basic healing. Count each "
        "saved ally once per tick, even if several Priests helped. One save can "
        "have Basic and Ultimate help, so those shares can add up to more than 100%.",
        scopes=("team",),
        numerator="unique saves with useful Basic healing from this team",
        denominator="all unique saves by this team, from Basic and Ultimate healing",
        direction="descriptive",
        subject_role="team",
    )

    for outcome, family, denominator in (
        ("kill", "kill_contributions", "all kills by this team"),
        ("rescue", "priest_rescue", "all rescues by this team"),
    ):
        contribution_description = (
            (
                "How many kills this agent helped with by activating its Ultimate "
                "on the tick the enemy died. It must "
                "deal Ultimate damage, or give an attacker some useful Ultimate "
                "healing, on the tick the enemy dies. This "
                "does not mean the Ultimate was needed to get the kill. "
                + _BURST_KILL_NOTE
            )
            if outcome == "kill"
            else (
                "How many times this agent helped save an ally from death with "
                "Ultimate healing on that tick. At least part of the heal must "
                "be useful. Several Priests can help; one helper may not have had "
                "enough healing to save the ally alone."
            )
        )
        add(
            f"ultimate_{outcome}_contributions",
            f"Ultimate {outcome.title()} Contributions",
            family,
            "count",
            contribution_description,
            scopes=("agent",),
            direction="higher",
        )
        add(
            f"ultimate_{outcome}_participation",
            "Ultimate Kill Participation"
            if outcome == "kill"
            else "Share of Team Saves Helped by This Priest's Ultimate Healing",
            family,
            "fraction",
            (
                "What share of this team's kills this agent helped with by activating "
                "its Ultimate on the tick the enemy died. Several teammates can help "
                "with the same kill. " + _BURST_KILL_NOTE
                if outcome == "kill"
                else f"What share of this team's {outcome}s this agent helped with "
                "using its Ultimate. Several teammates can help with the same "
                f"{outcome}."
            ),
            scopes=("agent",),
            denominator=denominator,
            direction="descriptive",
        )
        add(
            f"ultimate_{outcome}s",
            "Kills Helped by an Ultimate Activated That Tick"
            if outcome == "kill"
            else f"{outcome.title()}s With Ultimate Contributions",
            family,
            "count",
            (
                "How many of this team's kills had help from an Ultimate activated "
                "on the tick the enemy died. Help means damaging the enemy or giving "
                "an attacker some useful healing. Each enemy death counts "
                "once, even if several teammates helped. "
                + _BURST_KILL_NOTE
                + " A teammate's Ultimate can still make the same kill count here."
                if outcome == "kill"
                else f"How many {outcome}s this team got with help from an Ultimate on "
                f"that tick. Each {outcome} counts once even if several teammates "
                "helped."
            ),
            scopes=("team",),
            direction="higher",
        )
        add(
            f"ultimate_{outcome}_fraction",
            "Share of Kills Helped by an Ultimate Activated That Tick"
            if outcome == "kill"
            else "Share of Saves Helped by Ultimate Healing",
            family,
            "fraction",
            (
                "What share of this team's kills had help from an Ultimate activated "
                "on the tick the enemy died. Help means damaging the enemy or giving "
                "an attacker some useful healing. "
                + _BURST_KILL_NOTE
                + " A teammate's Ultimate can still make the same kill count here."
                if outcome == "kill"
                else f"What share of this team's {outcome}s had help from an Ultimate. "
                "This shows Ultimate use, not whether using Ultimates more often "
                "is better."
            ),
            scopes=("team",),
            denominator=denominator,
            direction="descriptive",
        )

    add(
        "burst_damage",
        "Damage to Recipient While Burst Is Active",
        "burst",
        "health",
        (
            "How much damage the From Mage dealt to the To enemy while Burst "
            "was already active at the start of the tick. Includes damage "
            "beyond the enemy's remaining health. Turning on Burst deals no "
            "damage."
        ),
        scopes=("source_recipient",),
        pairs=enemy_recipients,
        required_class_id=1,
        direction="higher",
    )
    add(
        "ultimate_effective_healing_done",
        "Effective Ultimate Healing Done",
        "healing_done",
        "health",
        (
            "How much Ultimate healing {subject} provided, minus the part that "
            "was more than needed. Priests healing the same ally share that excess in "
            "proportion to how much each healed."
        ),
        direction="higher",
    )
    add(
        "ultimate_effective_healing_done",
        "Effective Ultimate Healing Done to Recipient",
        "recipient_healing",
        "health",
        (
            "How much Salvation healing this agent gave this target, minus the "
            "part that was more than needed. Priests healing the same ally share "
            "excess in proportion to how much each healed."
        ),
        scopes=("source_recipient",),
        pairs=ally_recipients,
        direction="higher",
    )
    add(
        "ultimate_effective_healing_fraction",
        "Effective Fraction of Ultimate Healing",
        "healing_done",
        "fraction",
        ("How much of the Ultimate healing {subject} provided was useful."),
        denominator="all Ultimate healing {subject} provided, including excess",
        direction="higher",
    )
    add(
        "ultimate_excess_healing_fraction",
        "Excess Fraction of Ultimate Healing",
        "excess_healing",
        "fraction",
        ("How much of the Ultimate healing {subject} provided was more than needed."),
        denominator="all Ultimate healing {subject} provided, including excess",
        direction="lower",
    )

    # Matrix measures share one fixed source/recipient inventory. Existing names
    # are references to the original definition, never renamed numerical aliases.
    for metric in DIRECTED_METRICS:
        ability = metric.key.removesuffix("_applications").title()
        is_ability_use = metric.family == "abilities"
        controlled = metric.family in (
            "controlled_damage",
            "controlled_healing",
            "controlled_kills",
        )
        controlled_measure = {
            "controlled_damage": "Damage",
            "controlled_healing": "Healing",
            "controlled_kills": "Kill Contributions",
        }.get(metric.family, "")
        controlled_target = (
            "Target With " + STATUS_LABELS[metric.status_channel]
            if controlled and metric.status_channel is not None
            else ""
        )
        pairs = RECIPIENT_PAIRS_BY_RELATION[metric.relation]
        if (
            metric.family == "controlled_healing" and metric.status_channel in (3, 4, 5)
        ) or (
            metric.family == "aura_coverage" and metric.key.endswith("eligible_steps")
        ):
            pairs = tuple(pair for pair in pairs if pair[0] != pair[1])
        team_recipients = tuple(
            sorted({(source // 5 + 1, recipient) for source, recipient in pairs})
        )
        output_family = (
            "damage_done"
            if metric.family == "recipient_damage"
            else "excess_healing"
            if "excess_healing" in metric.key
            else "healing_done"
            if metric.family == "recipient_healing"
            else metric.family
        )
        amount_direction: MetricDirection = (
            "descriptive"
            if metric.key == "damage_to_hunter_trap_recipient"
            or (
                "excess_healing" in metric.key
                and not metric.key.startswith("ultimate_")
            )
            else "lower"
            if "excess_healing" in metric.key
            else "higher"
            if (metric.subject_role == "contributor" and metric.family != "trap_breaks")
            or metric.family in ("recipient_damage", "controlled_damage")
            or metric.key.startswith("burst_damage")
            or "effective_healing" in metric.key
            else "descriptive"
        )
        add(
            metric.source_total,
            metric.label,
            output_family,
            metric.unit,
            _directed_description(metric, "agent"),
            scopes=("agent",),
            direction=amount_direction,
            required_class_id=metric.required_class_id,
            status_channel=metric.status_channel,
            subject_role=metric.subject_role,
            reuse=True,
        )
        add(
            metric.team_total,
            {
                "basic_kills": "Kills Helped by Basic Abilities",
                "burst_kills": "Kills Helped During Burst",
                "basic_rescues": "Rescues With Basic Healing",
            }.get(metric.team_total, metric.label),
            output_family,
            metric.unit,
            _directed_description(metric, "team"),
            scopes=("team",),
            direction=amount_direction,
            required_class_id=metric.required_class_id,
            status_channel=metric.status_channel,
            subject_role="team" if metric.shared_events else "source",
            reuse=True,
        )
        add(
            metric.amount,
            f"{controlled_measure} to {controlled_target}"
            if controlled
            else (
                f"Ticks This {'Mage' if metric.required_class_id == 1 else 'Warrior'} "
                + ("Covered" if "covered" in metric.key else "Could Cover")
                + " This Ally With Its Aura"
            )
            if metric.family == "aura_coverage"
            else metric.label + " on Recipient"
            if metric.key == "basic_applications"
            else metric.label + " to Recipient",
            metric.family,
            metric.unit,
            _directed_description(metric, "source_recipient"),
            scopes=() if metric.relation == "self" else ("source_recipient",),
            pairs=pairs,
            direction=amount_direction,
            required_class_id=metric.required_class_id,
            status_channel=metric.status_channel,
            subject_role=metric.subject_role,
            recipient_role=metric.recipient_role,
            requires_basic_target=metric.requires_basic_target,
            requires_ultimate_target=metric.requires_ultimate_target,
            reuse=True,
        )
        recipient_family = (
            "damage_received"
            if metric.family == "recipient_damage"
            else "excess_healing"
            if "excess_healing" in metric.key
            else "healing_received"
            if metric.family == "recipient_healing"
            else metric.family
        )
        recipient_class = (
            metric.required_class_id
            if metric.recipient_scope == "team_recipient"
            else None
        )
        recipient_description = _directed_description(
            metric, metric.recipient_scope, recipient=True
        )
        recipient_direction: MetricDirection = (
            "descriptive"
            if metric.key == "damage_to_hunter_trap_recipient"
            else "lower"
            if metric.relation == "enemy" and amount_direction == "higher"
            else amount_direction
        )
        add(
            metric.recipient_total,
            f"{ability} Ability Activations Received"
            if is_ability_use
            else (
                "Deaths While This Effect Was Active"
                if metric.family == "controlled_kills"
                else ("Damage" if metric.family == "controlled_damage" else "Healing")
                + " Received While This Effect Was Active"
            )
            if controlled
            else {
                "rescues": "Times Rescued",
                "trap_breaks": "Times a Trap on This Agent Was Broken",
                "burst_damage_received": "Damage Received From Mages With Burst",
                "burst_damage_contributing_to_kill_received": (
                    "Burst Damage Received on Death Ticks"
                ),
                "mage_aura_covered_recipient_steps": "Ticks Covered by Mage Aura",
                "warrior_aura_covered_recipient_steps": "Ticks Covered by Warrior Aura",
                "mage_aura_eligible_recipient_steps": (
                    "Ticks This Agent Could Receive Mage Aura"
                ),
                "warrior_aura_eligible_recipient_steps": (
                    "Ticks This Agent Could Receive Warrior Aura"
                ),
            }.get(
                metric.recipient_total,
                metric.recipient_total.replace("_", " ").title(),
            ),
            recipient_family,
            metric.unit,
            recipient_description,
            scopes=() if metric.relation == "self" else (metric.recipient_scope,),
            pairs=team_recipients,
            direction=recipient_direction,
            required_class_id=recipient_class,
            status_channel=metric.status_channel,
            subject_role="recipient",
            recipient_role=metric.recipient_role,
            requires_basic_target=metric.requires_basic_target
            if metric.recipient_scope == "team_recipient"
            else False,
            requires_ultimate_target=metric.requires_ultimate_target
            if metric.recipient_scope == "team_recipient"
            else False,
            reuse=True,
        )
        quantity = metric.label.removesuffix(" Done").lower()
        pair_quantity = f"This agent's {quantity} to this target"
        allocation_denominator = f"all {quantity} this agent provided"
        contribution_denominator = (
            f"all the team's {quantity} to this target, including this agent's"
        )
        shared_explanation = ""
        allocation_description = contribution_description = ""
        if metric.family in ("recipient_damage", "recipient_healing"):
            channel = (
                "Basic "
                if metric.key.startswith("basic_")
                else "Ultimate "
                if metric.key.startswith("ultimate_")
                else ""
            )
            damage = metric.family == "recipient_damage"
            measure = "damage" if damage else "healing"
            verb = "dealt" if damage else "gave"
            portion = (
                " that was useful"
                if "effective" in metric.key
                else " that was more than needed"
                if "excess" in metric.key
                else ""
            )
            pair_quantity = (
                f"The {channel}{measure} this agent {verb} this target{portion}"
            )
            allocation_denominator = (
                f"all {channel}{measure} this agent {verb}{portion}"
            )
            contribution_denominator = (
                f"all {channel}{measure} the team {verb} this target{portion}, "
                "including this agent's"
            )
            if channel == "Ultimate " and damage:
                contribution_denominator += " and every other class's Ultimate damage"
            allocation_description = (
                f"How much of the {channel}{measure} this agent {verb}{portion} "
                "went to this target."
            )
            contribution_description = (
                f"How much of the {channel}{measure} this team {verb} this "
                f"target{portion} came from this agent."
            )
            if channel == "Ultimate " and damage:
                contribution_description += (
                    " The team total includes every class's direct Ultimate damage, "
                    "not just this agent's class."
                )
        elif metric.family == "burst" and not metric.shared_events:
            on_kill_tick = (
                " on the tick the damaged enemy died"
                if metric.key.endswith("kill")
                else ""
            )
            pair_quantity = (
                "Damage this Mage dealt to this enemy with Burst already active "
                f"at the start of the tick{on_kill_tick}"
            )
            allocation_denominator = (
                "all damage this Mage dealt with Burst already active at the "
                "start of the tick"
                + (" to any enemy on the tick that enemy died" if on_kill_tick else "")
            )
            contribution_denominator = (
                "all damage this enemy took from this team's Mages with Burst "
                "already active at the start of the tick"
                f"{on_kill_tick}"
                ", including this Mage's"
            )
            allocation_description = (
                "How much of this Mage's Burst damage hit this enemy."
            )
            contribution_description = (
                "How much of this team's Burst damage to this enemy came from "
                "this Mage."
            )
        elif metric.family == "status_applications":
            status = STATUS_LABELS[metric.status_channel or 0]
            pair_quantity = (
                f"times this agent used the ability that gives {status} on this target"
            )
            allocation_denominator = "all this agent's uses of that ability"
            contribution_denominator = (
                "all this team's uses of that ability on this target, "
                "including this agent's"
            )
            allocation_description = (
                f"How much of this agent's use of the ability that gives {status} "
                "was on this target."
            )
            contribution_description = (
                f"How much of this team's use of the ability that gives {status} "
                "on this target came from this agent."
            )
        elif metric.family == "aura_coverage":
            aura = "Mage" if metric.required_class_id == 1 else "Warrior"
            pair_quantity = f"Ticks this agent's {aura} aura covered this ally"
            allocation_denominator = (
                "all this aura giver's covered ally-ticks, where each covered ally "
                "on each tick counts once"
            )
            contribution_denominator = (
                f"ticks at least one {aura} aura from this team covered this ally"
            )
            allocation_description = (
                "How much of this agent's aura coverage went to this ally."
            )
            contribution_description = (
                "How often this agent supplied a "
                f"{aura}"
                " aura while this ally was covered by one."
            )
            shared_explanation = (
                "Several auras can cover the ally on the same tick, so their "
                "shares can add up to more than 1 (100%)."
            )
        elif metric.shared_events:
            rescue = metric.family == "priest_rescue"
            trap = metric.family == "trap_breaks"
            channel = (
                " using Basic healing"
                if rescue and metric.key.startswith("basic_")
                else " using Ultimate healing"
                if rescue and metric.key.startswith("ultimate_")
                else " using a Basic ability"
                if metric.key.startswith("basic_")
                else " using an Ultimate"
                if metric.key.startswith("ultimate_")
                else " while Burst was already active at the start of the tick"
                if metric.key.startswith("burst_")
                else ""
            )
            if rescue:
                pair_quantity = f"Rescues this agent helped with for this ally{channel}"
                allocation_denominator = f"all rescues this agent helped with{channel}"
                contribution_denominator = "all times this ally was rescued"
                event = "rescue"
            elif trap:
                pair_quantity = "Traps this agent helped break on this enemy"
                allocation_denominator = "all Traps on enemies this agent helped break"
                contribution_denominator = "all times damage broke this enemy's Trap"
                event = "Trap break"
            else:
                pair_quantity = (
                    f"Kills this agent helped with against this enemy{channel}"
                )
                allocation_denominator = f"all kills this agent helped with{channel}"
                contribution_denominator = "all times this enemy died"
                event = "kill"
                if metric.status_channel is not None:
                    status = STATUS_LABELS[metric.status_channel]
                    condition = f"already had {status} at the start of the tick"
                    pair_quantity += f" while the enemy {condition}"
                    allocation_denominator += f" against enemies who {condition}"
                    contribution_denominator += f" while it {condition}"
            shared_explanation = (
                f"Several teammates can help with the same {event}, so their "
                "shares can add up to more than 1 (100%)."
            )
            if metric.key == "solo_kills":
                pair_quantity = "Solo kills this agent got against this enemy"
                allocation_denominator = "all solo kills this agent got"
                shared_explanation = "A solo kill has only one helper."
            target = "ally" if rescue else "enemy"
            allocation_description = (
                f"What share of the {event}s this agent helped with{channel} "
                f"involved this {target}."
            )
            contribution_description = (
                f"How often this agent helped when this {target} "
                + (
                    "was rescued."
                    if rescue
                    else "had a Trap broken."
                    if trap
                    else "died."
                )
            )
            if metric.key == "solo_kills":
                allocation_description = (
                    "What share of this agent's solo kills came from this enemy."
                )
                contribution_description = (
                    "How often this enemy's death was a solo kill by this agent."
                )
            if metric.status_channel is not None:
                status = STATUS_LABELS[metric.status_channel]
                allocation_description = (
                    f"What share of this agent's kills against enemies with {status} "
                    "involved this enemy."
                )
                contribution_description = (
                    f"How often this agent helped when this enemy died with {status}."
                )
                shared_explanation = "Their shares can add up to more than 1 (100%)."
        elif metric.family in ("controlled_damage", "controlled_healing"):
            effect = (
                "damage" if metric.family == "controlled_damage" else "Priest healing"
            )
            verb = "dealt" if effect == "damage" else "gave"
            status = STATUS_LABELS[metric.status_channel or 0]
            condition = f"already had {status} at the start of the tick"
            pair_quantity = (
                f"{effect.capitalize()} this agent {verb} this target "
                f"while it {condition}"
            )
            allocation_denominator = (
                f"all {effect} this agent {verb} to agents who {condition}"
            )
            contribution_denominator = (
                f"all {effect} the team {verb} this target while it {condition}, "
                "including this agent's"
            )
            allocation_description = (
                f"How much of this agent's {effect} to "
                f"{'enemies' if effect == 'damage' else 'allies'} with {status} "
                "went to this target."
            )
            contribution_description = (
                f"How much of this team's {effect} to this target while it had "
                f"{status} came from this agent."
            )
        if metric.allocation is not None and metric.relation != "self":
            add(
                metric.allocation,
                f"{ability} Ability Activation Allocation"
                if is_ability_use
                else _allocation_label(metric),
                metric.family,
                "fraction",
                f"How much of this agent's {ability} ability use was on this target. "
                "For example, 4 of its 10 activations gives 4/10 = 0.4 (40%)."
                if is_ability_use
                else allocation_description
                + " "
                + metric.description
                + " Example: 4 out of 10 gives 0.4, or 40%.",
                scopes=("source_recipient",),
                pairs=pairs,
                direction="descriptive",
                numerator=f"times this agent used its {ability} ability on this target"
                if is_ability_use
                else pair_quantity,
                denominator=f"all {ability} ability activations by this agent"
                if is_ability_use
                else allocation_denominator,
                required_class_id=metric.required_class_id,
                status_channel=metric.status_channel,
                subject_role=metric.subject_role,
                recipient_role=metric.recipient_role,
                requires_basic_target=metric.requires_basic_target,
                requires_ultimate_target=metric.requires_ultimate_target,
                reuse=True,
            )
        if metric.contribution is not None and metric.relation != "self":
            event_label = ""
            if metric.family == "aura_coverage":
                aura = "Mage" if metric.required_class_id == 1 else "Warrior"
                event_label = (
                    f"Share of This Ally's {aura} Aura Ticks Supplied by This Agent"
                )
            if metric.shared_events and metric.family in (
                "kill_contributions",
                "priest_rescue",
                "trap_breaks",
                "burst",
            ):
                ability_help = (
                    "Basic "
                    if metric.key.startswith("basic_")
                    else "Ultimate "
                    if metric.key.startswith("ultimate_")
                    else "Burst "
                    if metric.key.startswith("burst_")
                    else ""
                )
                event_label = (
                    "Share of This Enemy's Deaths From This Agent's Solo Kills"
                    if metric.key == "solo_kills"
                    else "Share of This Enemy's Trap Breaks Helped by This Agent"
                    if metric.family == "trap_breaks"
                    else "Share of This Ally's Saves With "
                    + ability_help
                    + "Help From This Agent"
                    if metric.family == "priest_rescue"
                    else "Share of This Enemy's Deaths With "
                    + ability_help
                    + "Help From This Agent"
                )
            add(
                metric.contribution,
                f"Contribution to Team {ability} Ability Activations on Recipient"
                if is_ability_use
                else f"Participation in Kills of {controlled_target}"
                if metric.family == "controlled_kills"
                else f"Share of Team {controlled_measure} to {controlled_target}"
                if controlled
                else event_label
                if event_label
                else (
                    "Participation in " if metric.shared_events else "Fraction of Team "
                )
                + metric.label
                + " to Recipient",
                metric.family,
                "fraction",
                f"How much of this team's {ability} ability use on this target "
                "came from this agent. The team total includes every class. "
                "For example, 4 of the team's 8 activations "
                "gives 4/8 = 0.5 (50%)."
                if is_ability_use
                else contribution_description
                + " "
                + metric.description
                + " "
                + shared_explanation,
                scopes=("source_recipient",),
                pairs=tuple(pair for pair in pairs if pair[0] != pair[1])
                if metric.family == "aura_coverage"
                else pairs,
                direction="descriptive",
                numerator=f"times this agent used its {ability} ability on this target"
                if is_ability_use
                else pair_quantity,
                denominator=(
                    f"all {ability} ability activations by this team on this target, "
                    "including every class"
                )
                if is_ability_use
                else contribution_denominator,
                required_class_id=metric.required_class_id,
                status_channel=metric.status_channel,
                subject_role=metric.subject_role,
                recipient_role=metric.recipient_role,
                requires_basic_target=metric.requires_basic_target,
                requires_ultimate_target=metric.requires_ultimate_target,
                reuse=True,
            )

    for class_id, class_name in enumerate(ULTIMATE_CLASS_NAMES, 1):
        for fraction in (False, True):
            add(
                class_name + "_basic_kill" + ("_fraction" if fraction else "s"),
                ("Share of Team Kills" if fraction else "Kills")
                + f" Helped by {class_name.title()} Basic Abilities",
                "kill_contributions",
                "fraction" if fraction else "count",
                ("What share" if fraction else "How many")
                + f" of this team's kills its {class_name.title()}s helped "
                "with using their Basic abilities on the tick the enemy died. "
                "Count each enemy death once, even if several agents of this "
                "class helped. "
                + (
                    "A Priest helps by giving an attacker some useful healing."
                    if class_id == 5
                    else "Help must come from Basic damage."
                )
                + (
                    " A Mage's Basic attack during Burst counts here."
                    if class_id == 1
                    else ""
                )
                + " Different classes can help with the same kill, so their shares "
                "can add up to more than 100%.",
                scopes=("team",),
                direction="descriptive" if fraction else "higher",
                required_class_id=class_id,
                denominator="all kills by this team" if fraction else None,
                subject_role="source",
            )
    for class_id, class_name in enumerate(ULTIMATE_CLASS_NAMES[1:], 2):
        for fraction in (False, True):
            add(
                class_name + "_ultimate_kill" + ("_fraction" if fraction else "s"),
                ("Share of This Team's Kills" if fraction else "Kills")
                + " Helped by "
                + _ULTIMATE_NAMES[class_id - 1],
                "kill_contributions",
                "fraction" if fraction else "count",
                ("What share" if fraction else "How many")
                + f" of this team's kills its {class_name.title()}s helped "
                "with using their Ultimates on the kill tick. Count each enemy "
                "death once even if several agents of this class helped. "
                + (
                    "A Priest helps by giving an attacker some useful healing."
                    if class_id == 5
                    else "Help must come from direct Ultimate damage."
                ),
                scopes=("team",),
                direction="descriptive" if fraction else "higher",
                required_class_id=class_id,
                denominator="all kills by this team" if fraction else None,
                subject_role="source",
            )
    for class_id, class_name in ((1, "mage"), (2, "warrior")):
        add(
            f"{class_name}_aura_coverage",
            class_name.title() + " Aura Coverage",
            "aura_coverage",
            "fraction",
            (
                "How often this agent's aura covered this ally when both were "
                "alive and had no spawn shield. "
                "Checks the start of each tick. Each aura giver is measured "
                "separately, even when auras overlap."
            ),
            scopes=("source_recipient",),
            pairs=tuple(
                pair
                for pair in RECIPIENT_PAIRS_BY_RELATION["ally"]
                if pair[0] != pair[1]
            ),
            direction="descriptive",
            required_class_id=class_id,
            denominator=(
                "ticks when both the From and To agents were alive without spawn "
                "shields, including ticks out of range"
            ),
            subject_role="emitter",
            recipient_role="covered_ally",
            reuse=True,
        )

    for ability in ("", "basic_", "ultimate_"):
        for portion in ("effective", "excess"):
            stem = f"{ability}{portion}_healing_fraction"
            title = ability.replace("_", " ").title()
            label = f"{portion.title()} Fraction of {title}Healing"
            portion_text = (
                "was useful" if portion == "effective" else "was more than needed"
            )
            description = (
                f"How much of this agent's {title}healing to this target "
                f"{portion_text}. "
                "Priests healing the same ally share excess in proportion to "
                "how much each healed. Does not include regeneration."
            )
            if ability == "basic_":
                add(
                    stem,
                    label,
                    "healing_done" if portion == "effective" else "excess_healing",
                    "fraction",
                    "How much of the Basic healing {subject} provided "
                    f"{portion_text}.",
                    direction="descriptive",
                    denominator=(
                        "all Basic healing {subject} provided, including excess"
                    ),
                    reuse=True,
                )
            add(
                stem,
                label + " to Recipient",
                "recipient_healing",
                "fraction",
                description,
                scopes=("source_recipient",),
                pairs=RECIPIENT_PAIRS_BY_RELATION["ally"],
                direction=("higher" if portion == "effective" else "lower")
                if ability == "ultimate_"
                else "descriptive",
                denominator=(
                    f"all {title}healing this agent gave this target, including excess"
                ),
                subject_role="source",
                recipient_role="recipient",
                reuse=True,
            )

    # Red Zone stems are declared last: _measure_order ranks stems by first use,
    # so every older stem keeps its rank and the schema-14 order is unchanged.
    # A Red Zone death is counted once; it gives the enemy team 2 points, so
    # these counts never come from score changes.
    red_zone_missing = (
        "the Red Zone rule was not recorded, or the game uses the neutral task"
    )
    location_rule = "Location is checked when combat resolves, before movement."
    add(
        "red_zone_kills",
        "Red Zone Kills",
        "red_zone",
        "count",
        (
            "How many enemies this team killed inside the enemy team's Red Zone, "
            "near the enemy's spawn. Each enemy death counts once, even though "
            f"it gives this team 2 points. {location_rule}"
        ),
        scopes=("team",),
        direction="higher",
        missing=red_zone_missing,
    )
    add(
        "red_zone_kill_contributions",
        "Red Zone Kill Contributions",
        "red_zone",
        "count",
        (
            "How many Red Zone kills this agent helped with. Damage on the kill "
            "tick counts. Useful healing of an attacker on that tick also "
            "counts. Earlier damage does not count here. Each agent counts once "
            "per enemy death."
        ),
        scopes=("agent",),
        direction="descriptive",
        missing=red_zone_missing,
    )
    add(
        "red_zone_kill_participation",
        "Share of Team Red Zone Kills",
        "red_zone",
        "fraction",
        (
            "What share of this team's Red Zone kills this agent helped with. "
            "Each agent's share is at most 100%. Several teammates can help with "
            "the same kill, so their shares together can exceed 100%."
        ),
        scopes=("agent",),
        direction="descriptive",
        numerator="Red Zone kills this agent helped with during the selected period",
        denominator="all Red Zone kills by this team during the same period",
        missing=red_zone_missing,
    )
    add(
        "red_zone_deaths",
        "Red Zone Deaths",
        "red_zone",
        "count",
        (
            "How many times this agent died inside its team's Red Zone, near its "
            "own spawn. Each death counts once and gives the enemy team 2 "
            f"points. {location_rule}"
        ),
        team_description=(
            "How many times agents on this team died inside their own team's "
            "Red Zone, near their own spawn. Each death counts once and gives "
            f"the enemy team 2 points. {location_rule}"
        ),
        scopes=("team", "agent"),
        direction="lower",
        missing=red_zone_missing,
    )
    add(
        "red_zone_death_fraction",
        "Share of Team Red Zone Deaths",
        "red_zone",
        "fraction",
        (
            "What share of this team's Red Zone deaths were this agent's deaths. "
            "All teammates use the same team total and recorded period, so their "
            "shares add up to 100% when that total is greater than zero."
        ),
        scopes=("agent",),
        direction="descriptive",
        numerator="this agent's Red Zone deaths during the selected period",
        denominator="all Red Zone deaths on this team during the same period",
        missing=red_zone_missing,
    )

    return tuple(columns)


_columns = _build_columns()
_MEASURE_ORDER = _measure_order(list(_columns))
PRIORITY_METRIC_COLUMNS = tuple(column for column in _columns if column.priority)
METRIC_COLUMNS = PRIORITY_METRIC_COLUMNS + tuple(
    sorted((column for column in _columns if not column.priority), key=metric_csv_order)
)
del _columns
PRIORITY_METRIC_NAMES = tuple(column.name for column in PRIORITY_METRIC_COLUMNS)
FULL_METRIC_NAMES = tuple(column.name for column in METRIC_COLUMNS)
# Saved full-report column order for each readable current-format (host schema
# 2) scalar schema. Schema 14 is today's order without the 44 Red Zone columns;
# a later schema bump must add its own entry here. Keys decide which saved
# scalar versions the result readers accept.
FULL_METRIC_NAMES_BY_SCHEMA_VERSION: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {
        14: tuple(
            column.name for column in METRIC_COLUMNS if column.family != "red_zone"
        ),
        15: FULL_METRIC_NAMES,
    }
)
METRIC_COLUMNS_BY_NAME = MappingProxyType(
    {column.name: column for column in METRIC_COLUMNS}
)
FAMILY_COLUMN_COUNTS = MappingProxyType(
    Counter(column.family for column in METRIC_COLUMNS)
)


# Search phrases name the measurements below; they do not change their meaning.
_METRIC_SEARCH_RULES = (
    (
        ("damage dealt", "outgoing damage", "damage output"),
        (
            "basic_damage_done",
            "basic_damage_done_allocation_fraction",
            "basic_damage_done_contribution_fraction",
            "damage_done",
            "damage_done_allocation_fraction",
            "damage_done_fraction",
            "ultimate_damage_done",
            "ultimate_damage_done_allocation_fraction",
            "ultimate_damage_done_contribution_fraction",
        ),
    ),
    (
        ("damage taken", "incoming damage"),
        (
            "basic_damage_received",
            "damage_received",
            "damage_received_fraction",
            "ultimate_damage_received",
        ),
    ),
    (
        ("healing provided", "outgoing healing", "heal allies", "heal teammates"),
        (
            "basic_effective_healing_done",
            "basic_effective_healing_done_allocation_fraction",
            "basic_effective_healing_done_contribution_fraction",
            "basic_effective_healing_fraction",
            "basic_healing_done",
            "basic_healing_done_allocation_fraction",
            "basic_healing_done_contribution_fraction",
            "basic_excess_healing",
            "basic_excess_healing_allocation_fraction",
            "basic_excess_healing_contribution_fraction",
            "basic_excess_healing_fraction",
            "effective_healing_done",
            "effective_healing_done_allocation_fraction",
            "effective_healing_done_contribution_fraction",
            "effective_healing_fraction",
            "healing_done",
            "healing_done_allocation_fraction",
            "healing_done_fraction",
            "ultimate_effective_healing_done",
            "ultimate_effective_healing_done_allocation_fraction",
            "ultimate_effective_healing_done_contribution_fraction",
            "ultimate_effective_healing_fraction",
            "ultimate_healing_done",
            "ultimate_healing_done_allocation_fraction",
            "ultimate_healing_done_contribution_fraction",
            "ultimate_excess_healing",
            "ultimate_excess_healing_allocation_fraction",
            "ultimate_excess_healing_contribution_fraction",
            "ultimate_excess_healing_fraction",
            "excess_healing",
            "excess_healing_allocation_fraction",
            "excess_healing_contribution_fraction",
            "excess_healing_fraction",
        ),
    ),
    (
        ("overhealing", "overheal", "excess healing", "excess heals", "healing excess"),
        (
            "basic_excess_healing",
            "basic_excess_healing_allocation_fraction",
            "basic_excess_healing_contribution_fraction",
            "basic_excess_healing_fraction",
            "basic_excess_healing_received",
            "ultimate_excess_healing",
            "ultimate_excess_healing_allocation_fraction",
            "ultimate_excess_healing_contribution_fraction",
            "ultimate_excess_healing_fraction",
            "ultimate_excess_healing_received",
            "excess_healing",
            "excess_healing_allocation_fraction",
            "excess_healing_contribution_fraction",
            "excess_healing_fraction",
            "excess_healing_received",
            "excess_priest_healing_received_fraction",
        ),
    ),
    (
        (
            "regen",
            "regeneration",
            "automatic healing",
            "passive healing",
            "natural health recovery",
        ),
        ("regenerated_healing", "regeneration_healing_received_fraction"),
    ),
    (
        ("health recovered", "health recovery", "health restored"),
        ("regenerated_healing",),
    ),
    (
        ("healing efficiency", "heal efficiency"),
        (
            "basic_effective_healing_fraction",
            "basic_excess_healing_fraction",
            "effective_healing_fraction",
            "effective_priest_healing_received_fraction",
            "ultimate_effective_healing_fraction",
            "ultimate_excess_healing_fraction",
            "excess_healing_fraction",
            "excess_priest_healing_received_fraction",
        ),
    ),
    (
        ("useful healing", "effective heals", "healing that was useful"),
        (
            "basic_effective_healing_done",
            "basic_effective_healing_done_allocation_fraction",
            "basic_effective_healing_done_contribution_fraction",
            "basic_effective_healing_fraction",
            "basic_effective_priest_healing_received",
            "effective_healing_done",
            "effective_healing_done_allocation_fraction",
            "effective_healing_done_contribution_fraction",
            "effective_healing_fraction",
            "effective_priest_healing_received",
            "effective_priest_healing_received_fraction",
            "ultimate_effective_healing_done",
            "ultimate_effective_healing_done_allocation_fraction",
            "ultimate_effective_healing_done_contribution_fraction",
            "ultimate_effective_healing_fraction",
            "ultimate_effective_priest_healing_received",
        ),
    ),
    (
        ("incoming healing", "healing taken"),
        (
            "basic_effective_priest_healing_received",
            "basic_priest_healing_received",
            "effective_healing_received",
            "effective_priest_healing_received",
            "effective_priest_healing_received_fraction",
            "healing_received",
            "healing_received_fraction",
            "priest_healing_received",
            "priest_healing_received_fraction",
            "regenerated_healing",
            "regeneration_healing_received_fraction",
            "ultimate_effective_priest_healing_received",
            "ultimate_priest_healing_received",
            "excess_priest_healing_received_fraction",
        ),
    ),
    (
        ("damage amplification", "damage boost", "extra damage", "damage increase"),
        ("damage_from_mage_aura",),
    ),
    (
        ("damage reduction", "damage prevented", "damage mitigation", "damage blocked"),
        ("damage_prevented_by_warrior_aura",),
    ),
    (
        (
            "healing prevented",
            "healing reduction",
            "healing denied",
            "healing blocked by poison",
        ),
        ("healing_prevented_by_poison",),
    ),
    (
        ("same target", "shared targets", "attack the same enemy", "focus fire"),
        ("focus_fire_concentration", "focus_fire_steps"),
    ),
    (
        ("multiple attackers", "attack together", "multi attacker ticks"),
        ("focus_fire_concentration", "focus_fire_steps"),
    ),
    (
        ("shared kills", "joint kills", "multiple kill helpers"),
        ("multi_contributor_kill_fraction", "multi_contributor_kills"),
    ),
    (
        ("solo kills", "single kill helper"),
        (
            "single_contributor_kill_fraction",
            "single_contributor_kills",
            "solo_kill_fraction",
            "solo_kill_participation",
            "solo_kills",
            "solo_kills_allocation_fraction",
        ),
    ),
    (
        (
            "spread out",
            "spread apart",
            "stay together",
            "group up",
            "distance between allies",
            "distance between teammates",
            "team spread",
            "team spacing",
            "teammate spacing",
        ),
        ("ally_distance_mean", "ally_distance_observations"),
    ),
    (
        ("allies in aura range", "aura uptime", "time in aura range"),
        (
            "mage_aura_coverage",
            "mage_aura_covered_recipient_steps",
            "mage_aura_covered_steps",
            "warrior_aura_coverage",
            "warrior_aura_covered_recipient_steps",
            "warrior_aura_covered_steps",
        ),
    ),
    (
        ("slow duration", "time slowed", "slowed time"),
        (
            "hunter_basic_slow_active_steps",
            "rogue_poison_slow_active_steps",
            "warrior_charge_slow_active_steps",
        ),
    ),
    (
        ("stun duration", "time stunned", "stunned time"),
        (
            "hunter_trap_active_steps",
            "rogue_poison_stun_active_steps",
            "warrior_charge_stun_active_steps",
        ),
    ),
    (("trap duration", "time trapped", "trapped time"), ("hunter_trap_active_steps",)),
    (
        ("stun applications", "stuns applied"),
        (
            "hunter_trap_applications",
            "hunter_trap_applications_allocation_fraction",
            "hunter_trap_applications_contribution_fraction",
            "rogue_poison_stun_applications",
            "rogue_poison_stun_applications_allocation_fraction",
            "rogue_poison_stun_applications_contribution_fraction",
            "warrior_charge_stun_applications",
            "warrior_charge_stun_applications_allocation_fraction",
            "warrior_charge_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("buff applications", "apply buffs", "helpful effect applications"),
        (
            "mage_burst_applications",
            "priest_freedom_applications",
            "priest_freedom_applications_allocation_fraction",
            "priest_freedom_applications_contribution_fraction",
        ),
    ),
    (
        ("debuff applications", "apply debuffs", "negative effect applications"),
        (
            "hunter_basic_slow_applications",
            "hunter_basic_slow_applications_allocation_fraction",
            "hunter_basic_slow_applications_contribution_fraction",
            "hunter_trap_applications",
            "hunter_trap_applications_allocation_fraction",
            "hunter_trap_applications_contribution_fraction",
            "rogue_poison_anti_heal_applications",
            "rogue_poison_anti_heal_applications_allocation_fraction",
            "rogue_poison_anti_heal_applications_contribution_fraction",
            "rogue_poison_slow_applications",
            "rogue_poison_slow_applications_allocation_fraction",
            "rogue_poison_slow_applications_contribution_fraction",
            "rogue_poison_stun_applications",
            "rogue_poison_stun_applications_allocation_fraction",
            "rogue_poison_stun_applications_contribution_fraction",
            "warrior_charge_slow_applications",
            "warrior_charge_slow_applications_allocation_fraction",
            "warrior_charge_slow_applications_contribution_fraction",
            "warrior_charge_stun_applications",
            "warrior_charge_stun_applications_allocation_fraction",
            "warrior_charge_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("buff duration", "buff uptime", "time buffed"),
        ("mage_burst_active_steps", "priest_freedom_active_steps"),
    ),
    (
        ("debuff duration", "debuff uptime", "time debuffed"),
        (
            "hunter_basic_slow_active_steps",
            "hunter_trap_active_steps",
            "rogue_poison_anti_heal_active_steps",
            "rogue_poison_slow_active_steps",
            "rogue_poison_stun_active_steps",
            "warrior_charge_slow_active_steps",
            "warrior_charge_stun_active_steps",
        ),
    ),
    (
        ("damage to debuffed enemies", "damage to enemies with negative effects"),
        (
            "damage_received_while_hunter_basic_slow",
            "damage_received_while_hunter_trap",
            "damage_received_while_rogue_poison_anti_heal",
            "damage_received_while_rogue_poison_slow",
            "damage_received_while_rogue_poison_stun",
            "damage_received_while_warrior_charge_slow",
            "damage_received_while_warrior_charge_stun",
            "damage_to_hunter_basic_slow_recipient",
            "damage_to_hunter_basic_slow_recipient_allocation_fraction",
            "damage_to_hunter_basic_slow_recipient_contribution_fraction",
            "damage_to_hunter_trap_recipient",
            "damage_to_hunter_trap_recipient_allocation_fraction",
            "damage_to_hunter_trap_recipient_contribution_fraction",
            "damage_to_rogue_poison_anti_heal_recipient",
            "damage_to_rogue_poison_anti_heal_recipient_allocation_fraction",
            "damage_to_rogue_poison_anti_heal_recipient_contribution_fraction",
            "damage_to_rogue_poison_slow_recipient",
            "damage_to_rogue_poison_slow_recipient_allocation_fraction",
            "damage_to_rogue_poison_slow_recipient_contribution_fraction",
            "damage_to_rogue_poison_stun_recipient",
            "damage_to_rogue_poison_stun_recipient_allocation_fraction",
            "damage_to_rogue_poison_stun_recipient_contribution_fraction",
            "damage_to_warrior_charge_slow_recipient",
            "damage_to_warrior_charge_slow_recipient_allocation_fraction",
            "damage_to_warrior_charge_slow_recipient_contribution_fraction",
            "damage_to_warrior_charge_stun_recipient",
            "damage_to_warrior_charge_stun_recipient_allocation_fraction",
            "damage_to_warrior_charge_stun_recipient_contribution_fraction",
        ),
    ),
    (
        ("healing debuffed allies", "healing allies with negative effects"),
        (
            "healing_received_while_hunter_basic_slow",
            "healing_received_while_hunter_trap",
            "healing_received_while_rogue_poison_anti_heal",
            "healing_received_while_rogue_poison_slow",
            "healing_received_while_rogue_poison_stun",
            "healing_received_while_warrior_charge_slow",
            "healing_received_while_warrior_charge_stun",
            "healing_to_hunter_basic_slow_recipient",
            "healing_to_hunter_basic_slow_recipient_allocation_fraction",
            "healing_to_hunter_basic_slow_recipient_contribution_fraction",
            "healing_to_hunter_trap_recipient",
            "healing_to_hunter_trap_recipient_allocation_fraction",
            "healing_to_hunter_trap_recipient_contribution_fraction",
            "healing_to_rogue_poison_anti_heal_recipient",
            "healing_to_rogue_poison_anti_heal_recipient_allocation_fraction",
            "healing_to_rogue_poison_anti_heal_recipient_contribution_fraction",
            "healing_to_rogue_poison_slow_recipient",
            "healing_to_rogue_poison_slow_recipient_allocation_fraction",
            "healing_to_rogue_poison_slow_recipient_contribution_fraction",
            "healing_to_rogue_poison_stun_recipient",
            "healing_to_rogue_poison_stun_recipient_allocation_fraction",
            "healing_to_rogue_poison_stun_recipient_contribution_fraction",
            "healing_to_warrior_charge_slow_recipient",
            "healing_to_warrior_charge_slow_recipient_allocation_fraction",
            "healing_to_warrior_charge_slow_recipient_contribution_fraction",
            "healing_to_warrior_charge_stun_recipient",
            "healing_to_warrior_charge_stun_recipient_allocation_fraction",
            "healing_to_warrior_charge_stun_recipient_contribution_fraction",
        ),
    ),
    (
        ("kills against debuffed enemies", "kill enemies with negative effects"),
        (
            "deaths_while_hunter_basic_slow",
            "deaths_while_hunter_trap",
            "deaths_while_rogue_poison_anti_heal",
            "deaths_while_rogue_poison_slow",
            "deaths_while_rogue_poison_stun",
            "deaths_while_warrior_charge_slow",
            "deaths_while_warrior_charge_stun",
            "kill_contributions_to_hunter_basic_slow_recipient",
            "kill_contributions_to_hunter_basic_slow_recipient_allocation_fraction",
            "kill_contributions_to_hunter_trap_recipient",
            "kill_contributions_to_hunter_trap_recipient_allocation_fraction",
            "kill_contributions_to_rogue_poison_anti_heal_recipient",
            "kill_contributions_to_rogue_poison_anti_heal_recipient_allocation_fraction",
            "kill_contributions_to_rogue_poison_slow_recipient",
            "kill_contributions_to_rogue_poison_slow_recipient_allocation_fraction",
            "kill_contributions_to_rogue_poison_stun_recipient",
            "kill_contributions_to_rogue_poison_stun_recipient_allocation_fraction",
            "kill_contributions_to_warrior_charge_slow_recipient",
            "kill_contributions_to_warrior_charge_slow_recipient_allocation_fraction",
            "kill_contributions_to_warrior_charge_stun_recipient",
            "kill_contributions_to_warrior_charge_stun_recipient_allocation_fraction",
            "kill_participation_in_hunter_basic_slow_recipient",
            "kill_participation_in_hunter_trap_recipient",
            "kill_participation_in_rogue_poison_anti_heal_recipient",
            "kill_participation_in_rogue_poison_slow_recipient",
            "kill_participation_in_rogue_poison_stun_recipient",
            "kill_participation_in_warrior_charge_slow_recipient",
            "kill_participation_in_warrior_charge_stun_recipient",
            "kills_of_hunter_basic_slow_recipient",
            "kills_of_hunter_trap_recipient",
            "kills_of_rogue_poison_anti_heal_recipient",
            "kills_of_rogue_poison_slow_recipient",
            "kills_of_rogue_poison_stun_recipient",
            "kills_of_warrior_charge_slow_recipient",
            "kills_of_warrior_charge_stun_recipient",
        ),
    ),
    (
        ("damage to slowed enemies",),
        (
            "damage_received_while_hunter_basic_slow",
            "damage_received_while_rogue_poison_slow",
            "damage_received_while_warrior_charge_slow",
            "damage_to_hunter_basic_slow_recipient",
            "damage_to_hunter_basic_slow_recipient_allocation_fraction",
            "damage_to_hunter_basic_slow_recipient_contribution_fraction",
            "damage_to_rogue_poison_slow_recipient",
            "damage_to_rogue_poison_slow_recipient_allocation_fraction",
            "damage_to_rogue_poison_slow_recipient_contribution_fraction",
            "damage_to_warrior_charge_slow_recipient",
            "damage_to_warrior_charge_slow_recipient_allocation_fraction",
            "damage_to_warrior_charge_slow_recipient_contribution_fraction",
        ),
    ),
    (
        ("damage to stunned enemies",),
        (
            "damage_received_while_hunter_trap",
            "damage_to_hunter_trap_recipient",
            "damage_to_hunter_trap_recipient_allocation_fraction",
            "damage_to_hunter_trap_recipient_contribution_fraction",
            "damage_received_while_rogue_poison_stun",
            "damage_received_while_warrior_charge_stun",
            "damage_to_rogue_poison_stun_recipient",
            "damage_to_rogue_poison_stun_recipient_allocation_fraction",
            "damage_to_rogue_poison_stun_recipient_contribution_fraction",
            "damage_to_warrior_charge_stun_recipient",
            "damage_to_warrior_charge_stun_recipient_allocation_fraction",
            "damage_to_warrior_charge_stun_recipient_contribution_fraction",
        ),
    ),
    (
        ("damage to trapped enemies",),
        (
            "damage_received_while_hunter_trap",
            "damage_to_hunter_trap_recipient",
            "damage_to_hunter_trap_recipient_allocation_fraction",
            "damage_to_hunter_trap_recipient_contribution_fraction",
        ),
    ),
    (
        ("healing slowed allies",),
        (
            "healing_received_while_hunter_basic_slow",
            "healing_received_while_rogue_poison_slow",
            "healing_received_while_warrior_charge_slow",
            "healing_to_hunter_basic_slow_recipient",
            "healing_to_hunter_basic_slow_recipient_allocation_fraction",
            "healing_to_hunter_basic_slow_recipient_contribution_fraction",
            "healing_to_rogue_poison_slow_recipient",
            "healing_to_rogue_poison_slow_recipient_allocation_fraction",
            "healing_to_rogue_poison_slow_recipient_contribution_fraction",
            "healing_to_warrior_charge_slow_recipient",
            "healing_to_warrior_charge_slow_recipient_allocation_fraction",
            "healing_to_warrior_charge_slow_recipient_contribution_fraction",
        ),
    ),
    (
        ("healing stunned allies",),
        (
            "healing_received_while_hunter_trap",
            "healing_to_hunter_trap_recipient",
            "healing_to_hunter_trap_recipient_allocation_fraction",
            "healing_to_hunter_trap_recipient_contribution_fraction",
            "healing_received_while_rogue_poison_stun",
            "healing_received_while_warrior_charge_stun",
            "healing_to_rogue_poison_stun_recipient",
            "healing_to_rogue_poison_stun_recipient_allocation_fraction",
            "healing_to_rogue_poison_stun_recipient_contribution_fraction",
            "healing_to_warrior_charge_stun_recipient",
            "healing_to_warrior_charge_stun_recipient_allocation_fraction",
            "healing_to_warrior_charge_stun_recipient_contribution_fraction",
        ),
    ),
    (
        ("healing trapped allies",),
        (
            "healing_received_while_hunter_trap",
            "healing_to_hunter_trap_recipient",
            "healing_to_hunter_trap_recipient_allocation_fraction",
            "healing_to_hunter_trap_recipient_contribution_fraction",
        ),
    ),
    (
        ("kill slowed enemies",),
        (
            "deaths_while_hunter_basic_slow",
            "deaths_while_rogue_poison_slow",
            "deaths_while_warrior_charge_slow",
            "kill_contributions_to_hunter_basic_slow_recipient",
            "kill_contributions_to_hunter_basic_slow_recipient_allocation_fraction",
            "kill_contributions_to_rogue_poison_slow_recipient",
            "kill_contributions_to_rogue_poison_slow_recipient_allocation_fraction",
            "kill_contributions_to_warrior_charge_slow_recipient",
            "kill_contributions_to_warrior_charge_slow_recipient_allocation_fraction",
            "kill_participation_in_hunter_basic_slow_recipient",
            "kill_participation_in_rogue_poison_slow_recipient",
            "kill_participation_in_warrior_charge_slow_recipient",
            "kills_of_hunter_basic_slow_recipient",
            "kills_of_rogue_poison_slow_recipient",
            "kills_of_warrior_charge_slow_recipient",
        ),
    ),
    (
        ("kill stunned enemies",),
        (
            "deaths_while_hunter_trap",
            "kill_contributions_to_hunter_trap_recipient",
            "kill_contributions_to_hunter_trap_recipient_allocation_fraction",
            "kill_participation_in_hunter_trap_recipient",
            "kills_of_hunter_trap_recipient",
            "deaths_while_rogue_poison_stun",
            "deaths_while_warrior_charge_stun",
            "kill_contributions_to_rogue_poison_stun_recipient",
            "kill_contributions_to_rogue_poison_stun_recipient_allocation_fraction",
            "kill_contributions_to_warrior_charge_stun_recipient",
            "kill_contributions_to_warrior_charge_stun_recipient_allocation_fraction",
            "kill_participation_in_rogue_poison_stun_recipient",
            "kill_participation_in_warrior_charge_stun_recipient",
            "kills_of_rogue_poison_stun_recipient",
            "kills_of_warrior_charge_stun_recipient",
        ),
    ),
    (
        ("kill trapped enemies",),
        (
            "deaths_while_hunter_trap",
            "kill_contributions_to_hunter_trap_recipient",
            "kill_contributions_to_hunter_trap_recipient_allocation_fraction",
            "kill_participation_in_hunter_trap_recipient",
            "kills_of_hunter_trap_recipient",
        ),
    ),
    (
        (
            "save an ally",
            "save teammates",
            "healing rescues",
            "prevent a death",
            "save allies",
            "life saving healing",
            "clutch saves",
        ),
        (
            "basic_rescue_contributions",
            "basic_rescue_contributions_allocation_fraction",
            "basic_rescue_fraction",
            "basic_rescue_participation",
            "basic_rescues",
            "rescue_contributions",
            "rescue_contributions_allocation_fraction",
            "rescue_opportunities",
            "rescue_participation",
            "rescue_rate",
            "rescues",
            "ultimate_rescue_contributions",
            "ultimate_rescue_contributions_allocation_fraction",
            "ultimate_rescue_fraction",
            "ultimate_rescue_participation",
            "ultimate_rescues",
        ),
    ),
    (
        (
            "chances to save an ally",
            "rescue chances",
            "rescue opportunities",
            "save opportunities",
        ),
        ("rescue_opportunities",),
    ),
    (("save success rate", "rescue success rate"), ("rescue_rate",)),
    (
        (
            "help with kills",
            "kill participation",
            "kill assists",
            "kill credit",
            "kill involvement",
            "assist credit",
        ),
        (
            "basic_kill_contributions",
            "basic_kill_contributions_allocation_fraction",
            "basic_kill_fraction",
            "basic_kill_participation",
            "basic_kills",
            "hunter_basic_kill_fraction",
            "hunter_basic_kills",
            "hunter_ultimate_kill_fraction",
            "hunter_ultimate_kills",
            "kill_contributions",
            "kill_contributions_allocation_fraction",
            "kill_contributions_per_death",
            "kill_participation",
            "mage_basic_kill_fraction",
            "mage_basic_kills",
            "priest_basic_kill_fraction",
            "priest_basic_kills",
            "priest_ultimate_kill_fraction",
            "priest_ultimate_kills",
            "rogue_basic_kill_fraction",
            "rogue_basic_kills",
            "rogue_ultimate_kill_fraction",
            "rogue_ultimate_kills",
            "solo_kill_fraction",
            "solo_kill_participation",
            "solo_kills",
            "solo_kills_allocation_fraction",
            "ultimate_kill_contributions",
            "ultimate_kill_contributions_allocation_fraction",
            "ultimate_kill_fraction",
            "ultimate_kill_participation",
            "ultimate_kills",
            "warrior_basic_kill_fraction",
            "warrior_basic_kills",
            "warrior_ultimate_kill_fraction",
            "warrior_ultimate_kills",
        ),
    ),
    (
        ("return to life", "come back to life", "revive", "revival"),
        ("mean_agents_per_respawn_wave", "respawn_waves"),
    ),
    (
        ("respawn delay", "respawn waiting time", "waiting to respawn"),
        ("mean_observed_respawn_wait_steps",),
    ),
    (("death count", "repeated deaths"), ("deaths",)),
    (
        ("time spent dead", "dead time", "time dead"),
        ("dead_step_fraction", "dead_steps"),
    ),
    (
        ("ability use", "ability usage", "ability uses"),
        (
            "basic_activations",
            "basic_application_contribution_fraction",
            "basic_application_fraction",
            "basic_applications",
            "ultimate_activations",
            "ultimate_application_contribution_fraction",
            "ultimate_application_fraction",
            "ultimate_applications",
        ),
    ),
    (
        ("cast count", "casts", "times cast"),
        (
            "basic_activations",
            "basic_applications",
            "ultimate_activations",
            "ultimate_applications",
        ),
    ),
    (
        ("ability targets", "who targeted whom"),
        (
            "basic_application_contribution_fraction",
            "basic_application_fraction",
            "basic_applications",
            "ultimate_application_contribution_fraction",
            "ultimate_application_fraction",
            "ultimate_applications",
        ),
    ),
    (
        ("invalid actions", "illegal actions", "action rejection", "rejected choices"),
        (
            "action_combat_rejections",
            "action_domain_rejections",
            "action_movement_rejections",
            "actions_rejected",
        ),
    ),
    (
        ("legal actions", "allowed actions", "valid actions"),
        ("action_acceptance_rate", "actions_accepted"),
    ),
    (
        (
            "team reward",
            "episode return",
            "cumulative reward",
            "total reward",
            "reward sum",
        ),
        ("return",),
    ),
    (("game score", "match score", "team points"), ("score", "score_difference")),
    (
        ("match outcome", "game result", "match result", "game outcome"),
        ("draw", "loss", "win"),
    ),
    (
        (
            "slow protection",
            "movement speed protection",
            "weaken slows",
            "slow mitigation",
            "slow resistance",
        ),
        (
            "freedom_eligible_steps",
            "freedom_protected_steps",
            "freedom_protection_fraction",
        ),
    ),
    (
        ("break a trap", "trap breaks", "breaking traps", "traps broken by damage"),
        (
            "trap_break_contributions",
            "trap_break_contributions_allocation_fraction",
            "trap_break_participation",
            "trap_break_rate",
            "trap_breaks",
            "trap_intervals",
            "trap_mean_remaining_steps_at_break",
        ),
    ),
    (
        ("damage during burst", "damage while burst active"),
        (
            "burst_damage",
            "burst_damage_allocation_fraction",
            "burst_damage_contributing_to_kill",
            "burst_damage_contributing_to_kill_allocation_fraction",
            "burst_damage_contributing_to_kill_contribution_fraction",
            "burst_damage_contributing_to_kill_received",
            "burst_damage_contribution_fraction",
            "burst_damage_fraction",
            "burst_damage_received",
        ),
    ),
    (
        ("burst kill help", "burst kill participation", "burst kill assists"),
        (
            "burst_kill_contributions",
            "burst_kill_contributions_allocation_fraction",
            "burst_kill_participation",
            "burst_kills",
        ),
    ),
    (
        ("charge stun",),
        (
            "damage_received_while_warrior_charge_stun",
            "damage_to_warrior_charge_stun_recipient",
            "damage_to_warrior_charge_stun_recipient_allocation_fraction",
            "damage_to_warrior_charge_stun_recipient_contribution_fraction",
            "deaths_while_warrior_charge_stun",
            "healing_received_while_warrior_charge_stun",
            "healing_to_warrior_charge_stun_recipient",
            "healing_to_warrior_charge_stun_recipient_allocation_fraction",
            "healing_to_warrior_charge_stun_recipient_contribution_fraction",
            "kill_contributions_to_warrior_charge_stun_recipient",
            "kill_contributions_to_warrior_charge_stun_recipient_allocation_fraction",
            "kill_participation_in_warrior_charge_stun_recipient",
            "kills_of_warrior_charge_stun_recipient",
            "warrior_charge_stun_active_steps",
            "warrior_charge_stun_applications",
            "warrior_charge_stun_applications_allocation_fraction",
            "warrior_charge_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("charge slow",),
        (
            "damage_received_while_warrior_charge_slow",
            "damage_to_warrior_charge_slow_recipient",
            "damage_to_warrior_charge_slow_recipient_allocation_fraction",
            "damage_to_warrior_charge_slow_recipient_contribution_fraction",
            "deaths_while_warrior_charge_slow",
            "healing_received_while_warrior_charge_slow",
            "healing_to_warrior_charge_slow_recipient",
            "healing_to_warrior_charge_slow_recipient_allocation_fraction",
            "healing_to_warrior_charge_slow_recipient_contribution_fraction",
            "kill_contributions_to_warrior_charge_slow_recipient",
            "kill_contributions_to_warrior_charge_slow_recipient_allocation_fraction",
            "kill_participation_in_warrior_charge_slow_recipient",
            "kills_of_warrior_charge_slow_recipient",
            "warrior_charge_slow_active_steps",
            "warrior_charge_slow_applications",
            "warrior_charge_slow_applications_allocation_fraction",
            "warrior_charge_slow_applications_contribution_fraction",
        ),
    ),
    (
        ("poison slow",),
        (
            "damage_received_while_rogue_poison_slow",
            "damage_to_rogue_poison_slow_recipient",
            "damage_to_rogue_poison_slow_recipient_allocation_fraction",
            "damage_to_rogue_poison_slow_recipient_contribution_fraction",
            "deaths_while_rogue_poison_slow",
            "healing_received_while_rogue_poison_slow",
            "healing_to_rogue_poison_slow_recipient",
            "healing_to_rogue_poison_slow_recipient_allocation_fraction",
            "healing_to_rogue_poison_slow_recipient_contribution_fraction",
            "kill_contributions_to_rogue_poison_slow_recipient",
            "kill_contributions_to_rogue_poison_slow_recipient_allocation_fraction",
            "kill_participation_in_rogue_poison_slow_recipient",
            "kills_of_rogue_poison_slow_recipient",
            "rogue_poison_slow_active_steps",
            "rogue_poison_slow_applications",
            "rogue_poison_slow_applications_allocation_fraction",
            "rogue_poison_slow_applications_contribution_fraction",
        ),
    ),
    (
        ("poison stun",),
        (
            "damage_received_while_rogue_poison_stun",
            "damage_to_rogue_poison_stun_recipient",
            "damage_to_rogue_poison_stun_recipient_allocation_fraction",
            "damage_to_rogue_poison_stun_recipient_contribution_fraction",
            "deaths_while_rogue_poison_stun",
            "healing_received_while_rogue_poison_stun",
            "healing_to_rogue_poison_stun_recipient",
            "healing_to_rogue_poison_stun_recipient_allocation_fraction",
            "healing_to_rogue_poison_stun_recipient_contribution_fraction",
            "kill_contributions_to_rogue_poison_stun_recipient",
            "kill_contributions_to_rogue_poison_stun_recipient_allocation_fraction",
            "kill_participation_in_rogue_poison_stun_recipient",
            "kills_of_rogue_poison_stun_recipient",
            "rogue_poison_stun_active_steps",
            "rogue_poison_stun_applications",
            "rogue_poison_stun_applications_allocation_fraction",
            "rogue_poison_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("ultimate healing", "ultimate heals"),
        (
            "ultimate_effective_healing_done",
            "ultimate_effective_healing_done_allocation_fraction",
            "ultimate_effective_healing_done_contribution_fraction",
            "ultimate_effective_healing_fraction",
            "ultimate_effective_priest_healing_received",
            "ultimate_healing_done",
            "ultimate_healing_done_allocation_fraction",
            "ultimate_healing_done_contribution_fraction",
            "ultimate_priest_healing_received",
            "ultimate_excess_healing",
            "ultimate_excess_healing_allocation_fraction",
            "ultimate_excess_healing_contribution_fraction",
            "ultimate_excess_healing_fraction",
            "ultimate_excess_healing_received",
        ),
    ),
    (
        ("ultimate healing saves", "ultimate rescues"),
        (
            "ultimate_rescue_contributions",
            "ultimate_rescue_contributions_allocation_fraction",
            "ultimate_rescue_fraction",
            "ultimate_rescue_participation",
            "ultimate_rescues",
        ),
    ),
    (
        ("healing assists", "support assists"),
        (
            "priest_basic_kill_fraction",
            "priest_basic_kills",
            "priest_ultimate_kill_fraction",
            "priest_ultimate_kills",
        ),
    ),
    (
        ("antiheal", "anti heal", "anti healing"),
        (
            "healing_prevented_by_poison",
            "rogue_poison_anti_heal_active_steps",
            "rogue_poison_anti_heal_applications",
            "rogue_poison_anti_heal_applications_allocation_fraction",
            "rogue_poison_anti_heal_applications_contribution_fraction",
        ),
    ),
    (
        ("crowd control applications", "crowd control uses"),
        (
            "hunter_basic_slow_applications",
            "hunter_basic_slow_applications_allocation_fraction",
            "hunter_basic_slow_applications_contribution_fraction",
            "hunter_trap_applications",
            "hunter_trap_applications_allocation_fraction",
            "hunter_trap_applications_contribution_fraction",
            "rogue_poison_slow_applications",
            "rogue_poison_slow_applications_allocation_fraction",
            "rogue_poison_slow_applications_contribution_fraction",
            "rogue_poison_stun_applications",
            "rogue_poison_stun_applications_allocation_fraction",
            "rogue_poison_stun_applications_contribution_fraction",
            "warrior_charge_slow_applications",
            "warrior_charge_slow_applications_allocation_fraction",
            "warrior_charge_slow_applications_contribution_fraction",
            "warrior_charge_stun_applications",
            "warrior_charge_stun_applications_allocation_fraction",
            "warrior_charge_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("crowd control duration", "crowd control time"),
        (
            "hunter_basic_slow_active_steps",
            "hunter_trap_active_steps",
            "rogue_poison_slow_active_steps",
            "rogue_poison_stun_active_steps",
            "warrior_charge_slow_active_steps",
            "warrior_charge_stun_active_steps",
        ),
    ),
    (
        ("effect applications", "effects applied"),
        (
            "hunter_basic_slow_applications",
            "hunter_basic_slow_applications_allocation_fraction",
            "hunter_basic_slow_applications_contribution_fraction",
            "hunter_trap_applications",
            "hunter_trap_applications_allocation_fraction",
            "hunter_trap_applications_contribution_fraction",
            "mage_burst_applications",
            "priest_freedom_applications",
            "priest_freedom_applications_allocation_fraction",
            "priest_freedom_applications_contribution_fraction",
            "rogue_poison_anti_heal_applications",
            "rogue_poison_anti_heal_applications_allocation_fraction",
            "rogue_poison_anti_heal_applications_contribution_fraction",
            "rogue_poison_slow_applications",
            "rogue_poison_slow_applications_allocation_fraction",
            "rogue_poison_slow_applications_contribution_fraction",
            "rogue_poison_stun_applications",
            "rogue_poison_stun_applications_allocation_fraction",
            "rogue_poison_stun_applications_contribution_fraction",
            "warrior_charge_slow_applications",
            "warrior_charge_slow_applications_allocation_fraction",
            "warrior_charge_slow_applications_contribution_fraction",
            "warrior_charge_stun_applications",
            "warrior_charge_stun_applications_allocation_fraction",
            "warrior_charge_stun_applications_contribution_fraction",
        ),
    ),
    (
        ("status uptime", "effect duration", "time with effects"),
        (
            "hunter_basic_slow_active_steps",
            "hunter_trap_active_steps",
            "mage_burst_active_steps",
            "priest_freedom_active_steps",
            "rogue_poison_anti_heal_active_steps",
            "rogue_poison_slow_active_steps",
            "rogue_poison_stun_active_steps",
            "warrior_charge_slow_active_steps",
            "warrior_charge_stun_active_steps",
        ),
    ),
    (
        ("episode duration", "match length", "game length", "ticks played"),
        ("episode_length",),
    ),
    (("submitted actions", "chosen actions", "actions sent"), ("actions_submitted",)),
    (("invalid movement", "rejected movement"), ("action_movement_rejections",)),
    (
        ("invalid ability choice", "invalid target choice", "rejected combat choice"),
        ("action_combat_rejections",),
    ),
    (("unrecognized action", "unknown action choice"), ("action_domain_rejections",)),
    (
        ("trap time left at break", "remaining trap time at break"),
        ("trap_mean_remaining_steps_at_break",),
    ),
)


def _build_metric_search_terms() -> Mapping[str, tuple[str, ...]]:
    """Build immutable search aliases and reject aliases for unknown metric stems."""
    known_stems = {column.stem for column in METRIC_COLUMNS}
    by_stem: dict[str, list[str]] = {}
    for terms, stems in _METRIC_SEARCH_RULES:
        for stem in stems:
            if stem not in known_stems:
                raise ValueError(f"Search terms name an unknown measurement: {stem}")
            by_stem.setdefault(stem, []).extend(terms)
    return MappingProxyType(
        {stem: tuple(dict.fromkeys(terms)) for stem, terms in by_stem.items()}
    )


_METRIC_SEARCH_TERMS = _build_metric_search_terms()


def metric_search_terms(column: MetricColumn) -> tuple[str, ...]:
    """Return extra words that can find a catalog measurement.

    Parameters
    ----------
    column : MetricColumn
        Existing MetricColumn.

    Returns
    -------
    tuple[str, ...]
        Ordered alias tuple for its stem, or () when no extra aliases are needed.

    Notes
    -----
    Host-only lookup. Aliases affect search, not metric values or CSV identity.
    """
    return _METRIC_SEARCH_TERMS.get(column.stem, ())


_SEARCH_DIRECTED_PAIRS = MappingProxyType(
    {
        stem: metric
        for metric in DIRECTED_METRICS
        for stem in (metric.amount, metric.allocation, metric.contribution)
        if stem is not None
    }
)
_SEARCH_KIND_BY_FAMILY = MappingProxyType(
    {
        "abilities": "activation",
        "action_acceptance": "action",
        "damage_done": "damage",
        "recipient_damage": "damage",
        "damage_received": "damage",
        "controlled_damage": "damage",
        "healing_done": "healing",
        "recipient_healing": "healing",
        "healing_received": "healing",
        "controlled_healing": "healing",
        "excess_healing": "excess_healing",
        "priest_rescue": "rescue",
        "coordination": "coordination",
        "kill_contributions": "kill",
        "controlled_kills": "kill",
        "red_zone": "kill",
        "deaths": "death",
        "respawn": "respawn",
        "formation": "formation",
        "aura_coverage": "aura_coverage",
        "aura_benefits": "aura_benefit",
        "status_applications": "status_application",
        "status_active_steps": "status_time",
        "freedom": "freedom",
        "trap_breaks": "trap_break",
        "poison": "poison_prevention",
    }
)


def metric_search_facts(column: MetricColumn) -> dict[str, object]:
    """Describe a metric's search roles without guessing actual actors.

    Parameters
    ----------
    column : MetricColumn
        Existing MetricColumn with a known catalog family/stem.

    Returns
    -------
    dict[str, object]
        Dict with kind, ability, status, status subject, possible source/recipient
        classes, subject role, source kind, relation and qualifiers. None marks
        facts the static meaning does not establish.

    Notes
    -----
    Host-only metadata for a once-per-replay catalog, not per-tick computation.
    Source/recipient slots remain in scope/subjects. A class name inside a
    status identifies the effect, not its recorded caster; persistent status
    rows therefore keep unknown source identity where required.
    """
    stem, family, scope = column.stem, column.family, column.scope
    contexts = (
        (_SEARCH_DIRECTED_PAIRS[stem],)
        if scope == "source_recipient" and stem in _SEARCH_DIRECTED_PAIRS
        else _DIRECTED_CONTEXTS.get((scope, stem), ())
    )
    relations = {metric.relation for metric in contexts}
    relation = next(iter(relations)) if len(relations) == 1 else None
    if relation == "all":
        relation = None
    subject = (
        "episode"
        if scope == "episode"
        else "pair"
        if scope == "ally_pair"
        else "source"
        if scope in ("source_recipient", "team_recipient")
        else "recipient"
        if column.subject_role == "recipient"
        else "team"
        if scope == "team" and family == "priority"
        else "source"
    )
    if family == "priority":
        kind = (
            "outcome"
            if stem in ("win", "draw", "loss")
            else "episode"
            if stem == "episode_length"
            else "score"
            if stem in ("score", "score_difference")
            else "kill"
            if stem == "kills"
            else "death"
            if stem == "deaths"
            else "return"
        )
        if kind == "death":
            subject = "recipient"
        elif kind == "kill":
            subject = "source"
    elif family == "burst":
        kind = "damage" if "damage" in stem else "kill"
    else:
        kind = _SEARCH_KIND_BY_FAMILY[family]
    if family in ("controlled_kills", "red_zone") and subject == "recipient":
        kind = "death"
    if kind == "healing":
        if stem.startswith(("regenerated_", "regeneration_")):
            kind = "regeneration"
        elif "excess" in stem:
            kind = "excess_healing"
        elif "effective" in stem:
            kind = "effective_healing"
    if family == "coordination" and "kill" in stem:
        kind = "kill"

    ability = next(
        (
            name
            for name in ("basic", "ultimate", "burst")
            if stem.startswith(name + "_")
        ),
        None,
    )
    if family == "status_applications" and contexts:
        ability = "basic" if contexts[0].requires_basic_target else "ultimate"
    if stem in ULTIMATE_TEAM_ABILITY_STEMS.values():
        ability = "ultimate"
    if family == "kill_contributions" and column.required_class_id is not None:
        # Class-specific team kill counts still name their actual ability.
        ability = next(
            (name for name in ("basic", "ultimate") if f"_{name}_" in stem), ability
        )

    status = (
        STATUS_NAMES[column.status_channel]
        if column.status_channel is not None
        else "hunter_trap"
        if family == "trap_breaks"
        else "priest_freedom"
        if family == "freedom"
        else "rogue_poison_anti_heal"
        if family == "poison"
        else "mage_burst"
        if family == "burst"
        else None
    )
    source_class = column.required_class_id if subject != "recipient" else None
    recipient_class = column.required_class_id if subject == "recipient" else None
    if source_class is None and contexts:
        source_classes = {metric.required_class_id for metric in contexts}
        if len(source_classes) == 1:
            source_class = next(iter(source_classes))
    qualifiers: list[str] = [stem] if kind == "outcome" else []
    combined = family == "healing_received" and stem in (
        "healing_received",
        "healing_received_fraction",
        "effective_healing_received",
    )
    if combined:
        qualifiers.append("combined")
    elif kind in ("healing", "effective_healing", "excess_healing", "rescue"):
        source_class, relation = 5, "ally"
        qualifiers.append("priest")
    if family == "burst":
        source_class, relation = 1, "enemy"
    if family == "aura_coverage":
        relation = "ally"
        if contexts:
            source_class = contexts[0].required_class_id
    if kind in ("damage", "kill", "death", "trap_break"):
        relation = "enemy"
    elif family == "formation" or combined:
        relation = "ally"

    source = (
        "agent"
        if scope in ("agent", "source_recipient") and subject == "source"
        else "class"
        if subject == "recipient" and source_class is not None
        else "team"
    )
    if family in ("status_active_steps", "freedom", "poison"):
        source, source_class, relation = "unknown", None, None
    elif kind in (
        "regeneration",
        "formation",
        "episode",
        "outcome",
        "return",
        "score",
        "respawn",
    ):
        source, source_class = "none", None
    elif family == "aura_benefits":
        source, relation = "class", "ally"

    for word, qualifier in (
        ("allocation", "allocation"),
        ("contribution", "contribution"),
        ("participation", "participation"),
        ("fraction", "fraction"),
        ("solo", "solo"),
        ("single_contributor", "solo"),
        ("multi_contributor", "multi"),
        ("eligible", "eligible"),
        ("covered", "covered"),
        ("prevented", "prevented"),
        ("interval", "interval"),
        ("opportunities", "opportunity"),
        ("accepted", "accepted"),
        ("rejected", "rejected"),
        ("rejections", "rejected"),
        ("submitted", "submitted"),
    ):
        if word in stem and qualifier not in qualifiers:
            qualifiers.append(qualifier)
    if "contributing_to_kill" in stem:
        qualifiers.append("death_tick")
    if column.unit == "fraction" and "fraction" not in qualifiers:
        qualifiers.append("fraction")
    return {
        "kind": kind,
        "ability": ability,
        "status": status,
        "status_subject": (
            None if status is None else "source" if family == "burst" else "recipient"
        ),
        "source_class_id": source_class,
        "recipient_class_id": recipient_class,
        "subject": subject,
        "source": source,
        "relation": relation,
        "qualifiers": qualifiers,
    }
