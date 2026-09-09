"""Fixed scalar TDM measurements shared by numerical results, CSV and the UI.

This catalog describes columns; it performs no simulation or metric reduction.
Roster and run identity are separate row metadata. All ten slot names remain in
the schema even when a slot is inactive or contains another class.

Only damage/healing has source-to-recipient columns, and only formation has
unordered ally-pair columns. Team-to-recipient totals reuse ``damage_received``
and ``priest_healing_received``: a second spelling would duplicate those amounts.
Likewise, Burst activations reuse Ultimate activations, and poison duration reuses
the corresponding status-active steps. GUI groups may reference these columns.
"""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type MetricScope = Literal["episode", "team", "agent", "source_recipient", "ally_pair"]
type MetricDirection = Literal["higher", "lower", "descriptive"]

METRIC_SCHEMA_ID = "marlbg.tdm.scalar"
METRIC_SCHEMA_VERSION = 1

# Presentation belongs to the same catalog as scalar meaning; consumers need no
# private naming dictionary or guessed descriptions from snake_case identifiers.
METRIC_FAMILIES = MappingProxyType(
    {
        "overview": (
            "Team Overview",
            "Scores, outcomes, returns, and observed kills and deaths for "
            "both teams at the selected boundary.",
        ),
        "priority": (
            "Episode Results",
            "Episode length, team outcomes and scores, and canonical team "
            "and agent returns. Outcomes become available only when the "
            "game ends.",
        ),
        "abilities": (
            "Ability Activations",
            "Accepted Basic and Ultimate activations, counted from "
            "authoritative transition facts.",
        ),
        "deaths": (
            "Deaths and Time Dead",
            "Observed deaths and time spent dead across repeated lives, "
            "with each agent's share of team deaths.",
        ),
        "kill_contributions": (
            "Kill Contributions",
            "Lethal-tick damage and useful Priest healing of "
            "contributors, with one credit per agent, victim and tick.",
        ),
        "damage_done": (
            "Damage Done",
            "Delivered damage after source and recipient modifiers, "
            "before health clamping; includes overkill.",
        ),
        "recipient_damage": (
            "Damage by Recipient",
            "Delivered damage from each source to each enemy recipient, "
            "before health clamping.",
        ),
        "healing_done": (
            "Healing Done",
            "Delivered healing after modifiers, before health clamping; "
            "includes overheal and excludes regeneration.",
        ),
        "recipient_healing": (
            "Healing by Recipient",
            "Delivered healing from each source to each allied recipient; "
            "regeneration is excluded.",
        ),
        "damage_received": (
            "Damage Received",
            "Delivered enemy damage received by each agent and team, "
            "before health clamping.",
        ),
        "healing_received": (
            "Healing Received",
            "Delivered Priest healing, actual natural regeneration, and "
            "their shares of combined received healing.",
        ),
        "wasted_healing": (
            "Wasted Healing",
            "Priest healing lost to the combat upper health cap, "
            "attributed to the healing source and recipient; no regeneration.",
        ),
        "controlled_damage": (
            "Damage to Controlled Recipients",
            "Damage delivered to recipients with the named hostile status "
            "at transition start. Status conditions overlap; these are "
            "descriptive associations.",
        ),
        "controlled_healing": (
            "Healing to Controlled Recipients",
            "Healing delivered to recipients with the named hostile "
            "status at transition start. Status conditions overlap and "
            "regeneration is excluded.",
        ),
        "controlled_kills": (
            "Kills of Controlled Recipients",
            "Kills and contributions involving victims with the named "
            "hostile status at transition start. Overlapping status "
            "conditions are counted separately.",
        ),
        "coordination": (
            "Team Coordination",
            "Focus-fire concentration and single- or multiple-contributor "
            "kills describe how teammates coordinate their pressure.",
        ),
        "action_acceptance": (
            "Action Acceptance",
            "Submitted, fully accepted and rejected whole action tuples, "
            "with separate rejection reasons and authoritative dead-agent no-ops.",
        ),
        "status_applications": (
            "Status Applications",
            "Successful source-owned status applications, including "
            "refreshes; these are not distinct recipient status "
            "intervals.",
        ),
        "status_active_steps": (
            "Time Under Status",
            "Agent and team time under each named status, measured at "
            "transition-start decision boundaries.",
        ),
        "trap_breaks": (
            "Freezing Trap Breaks",
            "Observed Trap periods and damage-triggered breaks, including "
            "remaining duration and credited contributors.",
        ),
        "respawn": (
            "Respawn Waves",
            "Authoritative respawn waves and agents returned to play "
            "during the captured transitions.",
        ),
        "burst": (
            "Burst",
            "Mage damage and lethal-tick contributions while Burst is "
            "active, using the same delivered-damage accounting.",
        ),
        "aura_coverage": (
            "Aura Coverage",
            "Eligible ally time covered by Mage and Warrior auras. Team "
            "totals count each covered beneficiary once per tick.",
        ),
        "aura_benefits": (
            "Aura Benefits",
            "Counterfactual delivered-damage gains and prevention "
            "attributable to the active aura modifiers.",
        ),
        "poison": (
            "Crippling Poison",
            "Priest healing prevented at affected recipients by Rogue "
            "anti-heal; excludes regeneration and persistent caster attribution.",
        ),
        "priest_rescue": (
            "Priest Rescue",
            "Jointly feasible healing opportunities, successful rescues, "
            "and useful Priest participation in those rescues.",
        ),
        "freedom": (
            "Freedom",
            "Eligible agent time and time when Freedom raises the slowed "
            "movement floor; this does not measure actual distance gained.",
        ),
        "formation": (
            "Team Formation",
            "Distances between living teammates, summarized per team and "
            "unordered ally pair over eligible transitions.",
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
_STATUS_CLASS_IDS = (2, 3, 4, 2, 3, 4, 4, 1, 5)
_TEAMS = ((1,), (2,))
_AGENTS = tuple((slot,) for slot in range(10))
_ALLY_PAIRS = tuple(
    (first, second)
    for start in (0, 5)
    for first in range(start, start + 5)
    for second in range(first + 1, start + 5)
)


@dataclass(frozen=True, slots=True)
class MetricColumn:
    """One numeric measurement, with static interpretation and applicability.

    ``subjects`` contains a Core team ID, a global slot, or the ordered pair of
    global slots appropriate to ``scope``. A class requirement applies to the
    acting/emitting agent; for a team it requires that class in its active roster.
    Recipient-scoped effects do not imply knowledge of a persistent status caster.
    """

    name: str
    label: str
    family: str
    unit: str
    scope: MetricScope
    subjects: tuple[int, ...]
    description: str
    missing_when: str
    direction: MetricDirection = "descriptive"
    required_class_id: int | None = None
    status_channel: int | None = None
    priority: bool = False


def _prefix(scope: MetricScope, subjects: tuple[int, ...]) -> str:
    if scope == "episode":
        return ""
    if scope == "team":
        return "team_a_" if subjects[0] == 1 else "team_b_"
    if scope in ("source_recipient", "ally_pair"):
        return f"agent_{subjects[0]}_agent_{subjects[1]}_"
    return f"agent_{subjects[0]}_"


def _build_columns() -> tuple[MetricColumn, ...]:
    columns: list[MetricColumn] = []

    def add(
        name: str,
        label: str,
        family: str,
        unit: str,
        description: str,
        *,
        scopes: tuple[MetricScope, ...] = ("team", "agent"),
        pairs: tuple[tuple[int, int], ...] = (),
        denominator: str | None = None,
        required_class_id: int | None = None,
        status_channel: int | None = None,
        direction: MetricDirection = "descriptive",
        priority: bool = False,
        missing: str = "",
    ) -> None:
        for scope in scopes:
            subjects_by_scope = (
                ((),)
                if scope == "episode"
                else _TEAMS
                if scope == "team"
                else _AGENTS
                if scope == "agent"
                else pairs
            )
            unavailable = "Not collected"
            if scope != "episode":
                unavailable += "; subject or required recipient inactive"
            if required_class_id is not None:
                unavailable += "; required class absent from subject"
            if denominator is not None:
                unavailable += f"; {denominator} is zero"
            if missing:
                unavailable += f"; {missing}"
            for subjects in subjects_by_scope:
                columns.append(
                    MetricColumn(
                        name=_prefix(scope, subjects) + name,
                        label=label,
                        family=family,
                        unit=unit,
                        scope=scope,
                        subjects=subjects,
                        description=description,
                        missing_when=unavailable + ".",
                        direction=direction,
                        required_class_id=required_class_id,
                        status_channel=status_channel,
                        priority=priority,
                    )
                )

    # Priority order is also the numerical wrapper's fixed result-vector order.
    add(
        "episode_length",
        "Episode Length",
        "priority",
        "steps",
        "Number of real transitions, excluding terminal padding.",
        scopes=("episode",),
        priority=True,
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
                    description="One for this terminal outcome, otherwise zero.",
                    missing_when="Not collected; terminal outcome unavailable.",
                    direction="higher"
                    if outcome == "win"
                    else "lower"
                    if outcome == "loss"
                    else "descriptive",
                    priority=True,
                )
            )
    add(
        "return",
        "Return",
        "priority",
        "reward",
        "Sum of canonical rewards. Team return is the shared team reward, not the "
        "sum of its replicated agent rewards.",
        direction="higher",
        priority=True,
    )
    add(
        "score",
        "Score",
        "priority",
        "score",
        "Authoritative TDM score at the selected boundary, including any initialized "
        "scenario score; terminal for completed episode rows.",
        scopes=("team",),
        priority=True,
    )
    add(
        "score_difference",
        "Score Difference",
        "priority",
        "score",
        "Team A final score minus Team B final score.",
        scopes=("episode",),
        direction="higher",
        priority=True,
    )
    for name, label, description in (
        (
            "kills",
            "Total Kills",
            "Opposing newly-dead agents during real transitions; "
            "initialized score is excluded.",
        ),
        ("deaths", "Total Deaths", "Own newly-dead agents during real transitions."),
    ):
        add(
            name,
            label,
            "priority",
            "count",
            description,
            scopes=("team",),
            priority=True,
        )

    for ability in ("basic", "ultimate"):
        add(
            f"{ability}_activations",
            f"{ability.title()} Activations",
            "abilities",
            "count",
            "Accepted ability effects activated, including applications that "
            "refresh an existing status.",
        )

    add(
        "deaths",
        "Deaths",
        "deaths",
        "count",
        "New deaths of this agent.",
        scopes=("agent",),
    )
    add(
        "death_fraction",
        "Fraction of Team Deaths",
        "deaths",
        "fraction",
        "Agent deaths divided by its team's deaths.",
        scopes=("agent",),
        denominator="team deaths",
    )
    add(
        "dead_steps",
        "Steps Spent Dead",
        "deaths",
        "agent_steps",
        "Configured agents dead at transition start; team value sums agent time.",
    )
    add(
        "dead_step_fraction",
        "Fraction of Team Dead Time",
        "deaths",
        "fraction",
        "Agent dead steps divided by total team dead-agent steps.",
        scopes=("agent",),
        denominator="team dead-agent steps",
    )

    for name, label, unit, description, denominator in (
        (
            "kill_contributions",
            "Kill Contributions",
            "count",
            "Direct lethal-tick damage or useful Priest healing of a contributor, "
            "deduplicated per agent, victim and tick; no recursive support credit.",
            None,
        ),
        (
            "kill_contributions_per_death",
            "Kill Contributions per Death",
            "ratio",
            "Agent kill contributions divided by its deaths.",
            "agent deaths",
        ),
        (
            "kill_participation",
            "Kill Participation",
            "fraction",
            "Agent kill contributions divided by unique team kills.",
            "team kills",
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
        )

    for effect in ("damage", "healing"):
        add(
            f"{effect}_done",
            f"{effect.title()} Done",
            f"{effect}_done",
            "health",
            "Delivered amount after source and recipient modifiers, before health "
            "clamping; includes overkill/overheal and excludes regeneration.",
        )
        add(
            f"{effect}_done_fraction",
            f"Fraction of Team {effect.title()}",
            f"{effect}_done",
            "fraction",
            f"Agent {effect} done divided by its team's {effect} done.",
            scopes=("agent",),
            denominator=f"team {effect} done",
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
            "Delivered source-to-recipient amount from the shared effect matrix; "
            "includes overkill/overheal and excludes regeneration.",
            scopes=("source_recipient",),
            pairs=recipients,
        )
        add(
            f"{effect}_done_fraction",
            f"Fraction of Team {effect.title()} to Recipient",
            f"recipient_{effect}",
            "fraction",
            f"Source {effect} to this recipient divided by all same-team sources' "
            f"{effect} to that same recipient.",
            scopes=("source_recipient",),
            pairs=recipients,
            denominator=f"team {effect} to this recipient",
        )

    add(
        "damage_received",
        "Damage Received",
        "damage_received",
        "health",
        "Incoming delivered damage, including overkill. Agent values also provide "
        "the opposing team's damage to that recipient.",
    )
    add(
        "damage_received_fraction",
        "Fraction of Team Damage Received",
        "damage_received",
        "fraction",
        "Agent damage received divided by its team's damage received.",
        scopes=("agent",),
        denominator="team damage received",
    )
    for name, label, description in (
        (
            "priest_healing_received",
            "Priest Healing Received",
            "Delivered Priest healing, including overheal; also the team's healing "
            "to this recipient. Regeneration is excluded.",
        ),
        (
            "regenerated_healing",
            "Regenerated Healing",
            "Actual health restored by natural regeneration after combat.",
        ),
        (
            "healing_received",
            "Total Healing Received",
            "Delivered Priest healing plus actual natural regeneration.",
        ),
    ):
        add(name, label, "healing_received", "health", description)
    for source in ("priest", "regeneration"):
        add(
            f"{source}_healing_received_fraction",
            f"{source.title()} Fraction of Healing Received",
            "healing_received",
            "fraction",
            f"{source.title()} healing divided by combined delivered "
            "Priest healing and actual regeneration for the same subject.",
            denominator="combined healing received",
        )
    add(
        "healing_received_fraction",
        "Fraction of Team Healing Received",
        "healing_received",
        "fraction",
        "Agent combined healing received divided by its team's combined healing.",
        scopes=("agent",),
        denominator="team combined healing received",
    )

    add(
        "wasted_healing",
        "Wasted Healing",
        "wasted_healing",
        "health",
        "Priest healing lost to the combat upper health cap; simultaneous Priests "
        "share recipient waste in proportion to delivered healing. No regeneration.",
        required_class_id=5,
    )
    add(
        "wasted_healing_received",
        "Wasted Healing Received",
        "wasted_healing",
        "health",
        "Combat upper-cap Priest healing waste at this recipient, before "
        "natural regeneration.",
        scopes=("agent",),
    )

    for channel, status in enumerate(STATUS_NAMES[:7]):
        for effect in ("damage", "healing"):
            add(
                f"{effect}_to_{status}_recipient",
                f"{effect.title()} to Controlled Recipient",
                f"controlled_{effect}",
                "health",
                "Delivered amount to a recipient with this hostile status "
                "at transition start; statuses overlap and regeneration is excluded.",
                required_class_id=5 if effect == "healing" else None,
                status_channel=channel,
            )
        add(
            f"kills_of_{status}_recipient",
            "Kills of Controlled Recipient",
            "controlled_kills",
            "count",
            "Unique opposing deaths with this status "
            "at transition start, counted once per victim and tick.",
            scopes=("team",),
            status_channel=channel,
        )
        add(
            f"kill_contributions_to_{status}_recipient",
            "Controlled Kill Contributions",
            "controlled_kills",
            "count",
            "Agent's direct or useful Priest support "
            "contributions to opposing deaths with this transition-start status.",
            scopes=("agent",),
            status_channel=channel,
        )
        add(
            f"kill_participation_in_{status}_recipient",
            "Controlled Kill Participation",
            "controlled_kills",
            "fraction",
            "Agent controlled-kill contributions "
            "divided by unique team kills with the same status; agent fractions "
            "need not sum to one.",
            scopes=("agent",),
            denominator="team kills with this status",
            status_channel=channel,
        )

    for name, label, unit, description, denominator in (
        (
            "single_contributor_kills",
            "Single-Contributor Kills",
            "count",
            "Kills with exactly one contributor, including useful Priest support.",
            None,
        ),
        (
            "multi_contributor_kills",
            "Multiple-Contributor Kills",
            "count",
            "Kills with at least two contributors, including useful Priest support.",
            None,
        ),
        (
            "single_contributor_kill_fraction",
            "Single-Contributor Kill Fraction",
            "fraction",
            "Single-contributor kills divided by all unique team kills.",
            "team kills",
        ),
        (
            "multi_contributor_kill_fraction",
            "Multiple-Contributor Kill Fraction",
            "fraction",
            "Multiple-contributor kills divided by all unique team kills.",
            "team kills",
        ),
        (
            "focus_fire_concentration",
            "Focus Fire Concentration",
            "fraction",
            "Mean fraction of damaging agents sharing the most-targeted enemy, "
            "over ticks with at least two damaging agents; Priest support is excluded.",
            "focus-fire eligible ticks",
        ),
        (
            "focus_fire_steps",
            "Focus Fire Eligible Steps",
            "steps",
            "Real transitions with at least two damaging agents on the team.",
            None,
        ),
    ):
        add(
            name,
            label,
            "coordination",
            unit,
            description,
            scopes=("team",),
            denominator=denominator,
        )

    for name, label, description in (
        (
            "actions_submitted",
            "Actions Submitted",
            "One joint action tuple per "
            "configured agent per real transition, including dead-agent no-ops.",
        ),
        (
            "actions_accepted",
            "Actions Fully Accepted",
            "Submitted whole action tuples "
            "with no domain, movement or combat-pair rejection.",
        ),
        (
            "actions_rejected",
            "Actions Rejected",
            "Submitted whole action tuples "
            "with at least one rejection; a partially accepted tuple counts once.",
        ),
        (
            "action_domain_rejections",
            "Out-of-Domain Rejections",
            "Whole action "
            "tuples rejected because at least one submitted head is out of domain.",
        ),
        (
            "action_movement_rejections",
            "Movement Rejections",
            "In-domain movement rejections; may overlap combat-pair rejections.",
        ),
        (
            "action_combat_rejections",
            "Combat Rejections",
            "In-domain combat-pair rejections; may overlap movement rejections.",
        ),
    ):
        add(name, label, "action_acceptance", "count", description)
    add(
        "action_acceptance_rate",
        "Action Acceptance Rate",
        "action_acceptance",
        "fraction",
        "Fully accepted whole actions divided by submitted whole actions; "
        "individual rejection reasons are not additive.",
        denominator="actions submitted",
    )

    for channel, status in enumerate(STATUS_NAMES):
        add(
            f"{status}_applications",
            "Status Applications",
            "status_applications",
            "count",
            "Source-owned successful status applications, including refresh; "
            "not the number of distinct continuous recipient status intervals.",
            required_class_id=_STATUS_CLASS_IDS[channel],
            status_channel=channel,
        )
        add(
            f"{status}_active_steps",
            "Status Active Steps",
            "status_active_steps",
            "agent_steps",
            "Affected living agents with this status at transition "
            "start; team values sum recipient time, not persistent caster credit.",
            required_class_id=1 if channel == 7 else None,
            status_channel=channel,
        )

    for name, label, unit, description, denominator in (
        (
            "trap_intervals",
            "Observed Trap Periods",
            "count",
            "All observed continuous opposing Trap intervals, including initial and "
            "still-open intervals; refresh is not new, break then reapply is new.",
            None,
        ),
        (
            "trap_breaks",
            "Traps Broken by Damage",
            "count",
            "Unique opposing Trap intervals broken by accepted raw positive damage, "
            "using authoritative break facts after status ageing.",
            None,
        ),
        (
            "trap_break_rate",
            "Trap Break Rate",
            "fraction",
            "Damage-broken opposing Trap intervals divided by all observed opposing "
            "Trap intervals, including those still open.",
            "observed Trap intervals",
        ),
        (
            "trap_mean_remaining_steps_at_break",
            "Mean Trap Time Left at Break",
            "steps",
            "Mean remaining Trap timer at actual damage-break phase after ageing; "
            "natural expiry is excluded.",
            "damage-broken Trap intervals",
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
        )
    add(
        "trap_break_contributions",
        "Trap Break Contributions",
        "trap_breaks",
        "count",
        "Accepted raw positive damaging contributions to an authoritative opposing "
        "Trap break, deduplicated per source, recipient and tick.",
        scopes=("agent",),
    )
    add(
        "trap_break_participation",
        "Trap Break Participation",
        "trap_breaks",
        "fraction",
        "Agent Trap-break contributions divided by its team's unique "
        "damage breaks; agents may share credit.",
        scopes=("agent",),
        denominator="team damage-broken Trap intervals",
    )

    add(
        "respawn_waves",
        "Respawn Waves",
        "respawn",
        "count",
        "Authoritative team respawn-wave events.",
        scopes=("team",),
    )
    add(
        "mean_agents_per_respawn_wave",
        "Mean Agents per Respawn Wave",
        "respawn",
        "agents",
        "Agents respawned in team waves divided by wave count.",
        scopes=("team",),
        denominator="team respawn waves",
    )

    for name, label, unit, description, denominator in (
        (
            "burst_damage",
            "Burst Damage",
            "health",
            "Delivered Mage damage while Burst is active at transition start.",
            None,
        ),
        (
            "burst_damage_fraction",
            "Burst Fraction of Mage Damage",
            "fraction",
            "Burst-active damage divided by all damage from the same Mage subject; "
            "team denominator includes Mages only.",
            "Mage damage done",
        ),
        (
            "burst_damage_contributing_to_kill",
            "Burst Damage Contributing to Kill",
            "health",
            "Burst-active damage on a lethal tick to the newly dead recipient.",
            None,
        ),
        (
            "burst_kill_contributions",
            "Burst Kill Contributions",
            "count",
            "Direct kill contributions while Burst is active; team values sum Mage "
            "contributions rather than claim unique kills.",
            None,
        ),
    ):
        add(
            name,
            label,
            "burst",
            unit,
            description,
            denominator=denominator,
            required_class_id=1,
        )

    for aura, class_id in (("mage", 1), ("warrior", 2)):
        for suffix, label, unit, description, denominator in (
            (
                "covered_steps",
                "Aura-Covered Agent Steps",
                "agent_steps",
                "Covered eligible ally time. Emitter rows may overlap; team values "
                "count each covered beneficiary once per tick.",
                None,
            ),
            (
                "eligible_steps",
                "Aura-Eligible Agent Steps",
                "agent_steps",
                "Living unshielded ally time while the emitter is eligible; teams "
                "deduplicate beneficiaries when at least one eligible emitter exists.",
                None,
            ),
            (
                "coverage",
                "Aura Coverage",
                "fraction",
                "Covered eligible ally time "
                "divided by eligible ally time for that emitter or deduplicated team.",
                "aura-eligible agent steps",
            ),
        ):
            add(
                f"{aura}_aura_{suffix}",
                f"{aura.title()} {label}",
                "aura_coverage",
                unit,
                description,
                denominator=denominator,
                required_class_id=class_id,
            )
    add(
        "damage_from_mage_aura",
        "Damage Resulting from Mage Aura",
        "aura_benefits",
        "health",
        "Combined team damage gain from Mage auras, retaining Burst and "
        "recipient mitigation in the comparison; no arbitrary emitter split.",
        scopes=("team",),
        required_class_id=1,
    )
    add(
        "damage_prevented_by_warrior_aura",
        "Damage Prevented by Warrior Aura",
        "aura_benefits",
        "health",
        "Combined team incoming damage prevented by "
        "Warrior auras; no arbitrary emitter split.",
        scopes=("team",),
        required_class_id=2,
    )
    add(
        "healing_prevented_by_poison",
        "Healing Prevented by Rogue Poison",
        "poison",
        "health",
        "Priest healing prevented at this affected recipient/team by "
        "active Rogue anti-heal; no persistent caster attribution or regeneration.",
    )

    for name, label, unit, description, denominator in (
        (
            "rescue_opportunities",
            "Lethal Damage Rescue Opportunities",
            "count",
            "Lethally threatened ally-ticks where combined legally available Priest "
            "healing can save the ally; opportunities count recipients, not Priests.",
            None,
        ),
        (
            "rescues",
            "Lethal Damage Rescues",
            "count",
            "Threatened ally-ticks saved "
            "by Priest healing, counted once per saved ally and tick.",
            None,
        ),
        (
            "rescue_rate",
            "Lethal Damage Rescue Rate",
            "fraction",
            "Successful rescues divided by the team's available rescue opportunities.",
            "rescue opportunities",
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
        )
    add(
        "rescue_contributions",
        "Lethal Damage Rescue Contributions",
        "priest_rescue",
        "count",
        "Saved ally-ticks receiving this Priest's useful healing; shared rescue "
        "credit does not imply solo saving capability.",
        scopes=("agent",),
        required_class_id=5,
    )
    add(
        "rescue_participation",
        "Lethal Damage Rescue Participation",
        "priest_rescue",
        "fraction",
        "This Priest's useful-healing rescue contributions divided by unique team "
        "rescues; individual fractions may overlap.",
        scopes=("agent",),
        denominator="unique team rescues",
        required_class_id=5,
    )
    for name, label, unit, description, denominator in (
        (
            "freedom_protected_steps",
            "Freedom Slow Protection Steps",
            "agent_steps",
            "Eligible recipient-ticks where Freedom raises the applicable slowed "
            "movement floor; this is not actual distance gained.",
            None,
        ),
        (
            "freedom_eligible_steps",
            "Freedom Eligible Steps",
            "agent_steps",
            "Living unshielded unstunned recipient-ticks with Freedom active.",
            None,
        ),
        (
            "freedom_protection_fraction",
            "Freedom Slow Protection Fraction",
            "fraction",
            "Freedom-protected recipient time divided by Freedom-eligible time.",
            "Freedom-eligible agent steps",
        ),
    ):
        add(name, label, "freedom", unit, description, denominator=denominator)

    add(
        "ally_distance_mean",
        "Mean Ally Distance",
        "formation",
        "distance",
        "Mean start-of-transition distance over living unordered ally-pair "
        "observations; team values weight each eligible pair observation equally.",
        scopes=("team", "ally_pair"),
        pairs=_ALLY_PAIRS,
        denominator="living ally-pair observations",
    )
    add(
        "ally_distance_observations",
        "Ally Distance Observations",
        "formation",
        "pair_steps",
        "Number of living unordered ally-pair observations at "
        "transition start; no spawn-shield exclusion.",
        scopes=("team", "ally_pair"),
        pairs=_ALLY_PAIRS,
    )
    return tuple(columns)


METRIC_COLUMNS = _build_columns()
PRIORITY_METRIC_COLUMNS = tuple(column for column in METRIC_COLUMNS if column.priority)
PRIORITY_METRIC_NAMES = tuple(column.name for column in PRIORITY_METRIC_COLUMNS)
FULL_METRIC_NAMES = tuple(column.name for column in METRIC_COLUMNS)
METRIC_COLUMNS_BY_NAME = MappingProxyType(
    {column.name: column for column in METRIC_COLUMNS}
)
FAMILY_COLUMN_COUNTS = MappingProxyType(
    Counter(column.family for column in METRIC_COLUMNS)
)
