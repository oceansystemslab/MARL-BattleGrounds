"""Fixed-shape JAX collection for the scalar TDM metric catalog.

Only sufficient statistics survive a transition. Dictionary keys are static
PyTree structure, and all values are arrays; no replay, report or host callback
is required. Final and prefix analysis use this same collection authority.
"""

from functools import cache
from typing import cast

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import (
    ONLY_ALLY_TARGET_ULTIMATE_MODE,
    ONLY_ENEMY_TARGET_ULTIMATE_MODE,
    ONLY_NONE_TARGET_ULTIMATE_MODE,
    derive_effective_movement_speeds,
    get_ultimate_target_mode_by_class_ids,
)
from marl_battlegrounds.core.env import (
    _aggregate_health_effects_and_basic_passives_by_global_slot,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.core.types import (
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
)
from marl_battlegrounds.evaluation.combat_metrics import combat_quantities
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.metric_catalog import (
    DIRECTED_METRICS,
    METRIC_COLUMNS,
    PRIORITY_METRIC_COLUMNS,
    RECIPIENT_PAIRS_BY_RELATION,
    STATUS_CLASS_IDS,
    STATUS_NAMES,
    ULTIMATE_CLASS_NAMES,
    MetricScope,
)

type FullTotals = dict[str, Array]

_COUNTER_SHAPES = {
    "deaths": (10,),
    "dead_steps": (10,),
    "kills": (10, 10),
    "basic_applications": (10, 10),
    "actions": (10, 6),
    "active_steps": (10, 9),
    "controlled_kills": (10, 7),
    "controlled_contributions": (50, 7),
    "solo_kills": (50,),
    "multi_kills": (2,),
    "focus_steps": (2,),
    "trap_periods": (10,),
    "trap_breaks": (10,),
    "trap_remaining": (10,),
    "trap_contributions": (50,),
    "waves": (2,),
    "respawns": (10,),
    "ultimate_applications": (10, 10),
    "ultimate_kill_contributions": (50,),
    "ultimate_kills": (10,),
    "basic_kills": (10,),
    "class_basic_kills": (5, 2),
    "class_ultimate_kills": (5, 10),
    "ultimate_rescue_contributions": (50,),
    "ultimate_rescues": (10,),
    "basic_rescues": (10,),
    "burst_kills": (50,),
    "burst_unique_kills": (10,),
    "mage_coverage": (50,),
    "mage_eligible": (50,),
    "mage_team_coverage": (10,),
    "mage_team_eligible": (10,),
    "warrior_coverage": (50,),
    "warrior_eligible": (50,),
    "warrior_team_coverage": (10,),
    "warrior_team_eligible": (10,),
    "rescue_opportunities": (10,),
    "rescues": (10,),
    "rescue_contributions": (50,),
    "freedom_protected": (10,),
    "freedom_eligible": (10,),
    "distance_count": (10, 10),
}
_AMOUNT_SHAPES = {
    "damage": (10, 10),
    "healing": (10, 10),
    "excess": (10, 10),
    "basic_effective_healing": (50,),
    "basic_excess": (50,),
    "regeneration": (10,),
    "controlled_damage": (50, 7),
    "controlled_healing": (50, 7),
    "focus_sum": (2,),
    "burst_damage": (10, 10),
    "ultimate_damage": (10, 10),
    "ultimate_healing": (10, 10),
    "ultimate_excess": (10, 10),
    "burst_lethal_damage": (50,),
    "mage_damage_gain": (2,),
    "warrior_damage_prevented": (2,),
    "poison_prevention": (10,),
    "distance_sum": (10, 10),
}


def _team_sum(values: Array) -> Array:
    return values.reshape((2, 5, *values.shape[1:])).sum(axis=1)


def _pair_indices(relation: str) -> tuple[Array, Array]:
    pairs = RECIPIENT_PAIRS_BY_RELATION[relation]
    return (
        jnp.asarray(tuple(source for source, _ in pairs)),
        jnp.asarray(tuple(recipient for _, recipient in pairs)),
    )


def _pairs(matrix: Array, relation: str) -> Array:
    return matrix[_pair_indices(relation)]


def _matrix(pairs: Array, relation: str) -> Array:
    return (
        jnp.zeros((10, 10, *pairs.shape[1:]), pairs.dtype)
        .at[_pair_indices(relation)]
        .set(pairs)
    )


def initialize_full(config: EnvConfig, state: EnvState) -> FullTotals:
    """Retain initial dead flags and Trap periods; other evidence starts empty."""
    totals = {
        name: jnp.zeros(shape, jnp.int32) for name, shape in _COUNTER_SHAPES.items()
    }
    totals.update(
        {name: jnp.zeros(shape, jnp.float32) for name, shape in _AMOUNT_SHAPES.items()}
    )
    initial_traps = (
        config.agent_profile.active_mask
        & state.alive_mask
        & (state.stun_durations[:, 1] > 0)
    )
    totals["trap_periods"] = initial_traps.astype(jnp.int32)
    totals["initial_dead"] = config.agent_profile.active_mask & ~state.alive_mask
    return totals


def _aura_counts(
    config: EnvConfig, state: EnvState, coverage: Array, class_id: int
) -> tuple[Array, Array, Array, Array]:
    available = (
        config.agent_profile.active_mask
        & state.alive_mask
        & (state.spawn_shield_durations == 0)
    )
    emitter = available & (config.agent_profile.class_ids == class_id)
    team = config.agent_profile.team_ids
    eligible = emitter[:, None] & available[None, :] & (team[:, None] == team[None, :])
    return (
        _pairs(coverage, "ally"),
        _pairs(eligible, "ally"),
        coverage.any(axis=0),
        eligible.any(axis=0),
    )


def update_full(
    totals: FullTotals,
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    info: Info,
) -> FullTotals:
    """Collect one real transition; no-transition facts preserve all counters."""
    facts = info.transition_facts
    combat = facts.combat_transition_facts
    quantities = combat_quantities(config, state, action_mask, info)
    active = config.agent_profile.active_mask
    living = active & state.alive_mask
    damage = quantities.damage
    healing = quantities.healing
    deaths = facts.death_facts.is_newly_dead_by_recipient
    contributions = quantities.kill_contributions
    routed = quantities.routing
    statuses = (
        jnp.concatenate(
            (
                state.slow_durations,
                state.stun_durations,
                state.rogue_poison_anti_heal_durations[:, None],
                state.mage_burst_damage_amplification_durations[:, None],
                state.priest_blessing_of_freedom_slow_floor_durations[:, None],
            ),
            axis=1,
        )
        > 0
    )
    hostile_status = statuses[:, :7]
    acceptance = facts.action_acceptance_facts
    domain_rejection = acceptance.submitted_action_tuple_is_out_of_domain_by_actor
    move_rejection = acceptance.in_domain_move_action_is_rejected_by_actor
    combat_rejection = acceptance.in_domain_combat_action_pair_is_rejected_by_actor
    rejected = domain_rejection | move_rejection | combat_rejection
    damaging = damage > 0
    attacking_sources = damaging.any(axis=1).reshape(2, 5).sum(axis=1)
    most_shared_target = damaging.reshape(2, 5, 10).sum(axis=1).max(axis=1)
    focus_eligible = attacking_sources >= 2
    contributor_count = contributions.sum(axis=0)
    ultimate = combat.ultimate_effect_is_activated_by_source
    basic = combat.basic_effect_is_activated_by_source
    ultimate_contributions = contributions & ultimate[:, None]
    ultimate_rescues = quantities.rescue_contributions & ultimate[:, None]
    basic_contributions = contributions & basic[:, None]
    basic_rescues = quantities.rescue_contributions & basic[:, None]
    class_sources = config.agent_profile.class_ids[None, :] == jnp.arange(1, 6)[:, None]
    # Burst activates on its caster even though it has no directed combat effect.
    ultimate_routing = (
        jnp.where(
            (config.agent_profile.class_ids == 1)[:, None],
            jnp.eye(10, dtype=jnp.bool_),
            routed > 0,
        )
        & ultimate[:, None]
    )
    trap_breaks = (
        facts.status_lifecycle_facts.broken_by_damage_by_recipient_and_status_channel[
            :, 4
        ]
    )
    trap_applications = (
        routed * combat.stun_is_applied_by_source_and_channel[:, 1, None]
    ).sum(axis=0) > 0
    # A naturally expired or damage-broken period is closed before fresh control.
    trap_closed = (state.stun_durations[:, 1] <= 1) | trap_breaks
    new_traps = trap_applications & trap_closed
    burst = (state.mage_burst_damage_amplification_durations > 0) & (
        config.agent_profile.class_ids == 1
    )
    no_aura = _aggregate_health_effects_and_basic_passives_by_global_slot(
        state,
        config,
        acceptance.accepted_joint_action,
        routed,
        jnp.ones(10, jnp.float32),
        jnp.ones(10, jnp.float32),
    )
    mage_gain = (
        combat.source_modified_damage_output_by_source
        - no_aura.source_modified_damage_output_by_source
    ) * combat.recipient_damage_modifier_by_source
    prevented = (
        routed
        * (
            combat.source_modified_damage_output_by_source
            * (1 - combat.recipient_damage_modifier_by_source)
        )[:, None]
    )
    poison_prevention = (
        routed
        * (
            combat.source_modified_healing_output_by_source
            * (1 - combat.recipient_healing_modifier_by_source)
        )[:, None]
    )
    freedom = state.priest_blessing_of_freedom_slow_floor_durations
    freedom_eligible = (
        living
        & (state.spawn_shield_durations == 0)
        & ~(state.stun_durations > 0).any(axis=1)
        & (freedom > 0)
    )

    # Unit speed/scale compares the applicable restriction, not movement chosen.
    def restricted_speed(duration: Array) -> Array:
        return derive_effective_movement_speeds(
            state.slow_durations,
            duration,
            state.stun_durations,
            state.spawn_shield_durations,
            jnp.ones(10, jnp.float32),
            1.0,
            living,
            1.0,
        )

    freedom_protected = freedom_eligible & (
        restricted_speed(freedom) > restricted_speed(jnp.zeros_like(freedom))
    )
    same_team = (
        config.agent_profile.team_ids[:, None] == config.agent_profile.team_ids[None, :]
    )
    ally_pairs = (
        living[:, None]
        & living[None, :]
        & same_team
        & jnp.triu(jnp.ones((10, 10), bool), k=1)
    )
    distances = cast(
        Array,
        jnp.linalg.norm(
            state.agent_positions[:, None, :] - state.agent_positions[None, :, :],
            axis=-1,
        ),
    )
    delta: FullTotals = {
        "damage": damage,
        "healing": healing,
        "excess": quantities.excess_healing,
        # Subtracting independently rounded episode totals can invent effective
        # Basic healing or erase real excess. Retain these two channels directly.
        "basic_effective_healing": _pairs(
            (healing - quantities.excess_healing) * basic[:, None], "ally"
        ),
        "basic_excess": _pairs(quantities.excess_healing * basic[:, None], "ally"),
        "regeneration": (
            facts.regeneration_facts.actual_health_regenerated_this_step_by_agent
        ),
        "deaths": deaths,
        "dead_steps": active & ~state.alive_mask,
        "kills": contributions,
        "basic_applications": (routed > 0) & basic[:, None],
        "actions": jnp.stack(
            (
                active,
                active & ~rejected,
                active & rejected,
                active & domain_rejection,
                active & move_rejection,
                active & combat_rejection,
            ),
            axis=1,
        ),
        "active_steps": statuses & living[:, None],
        "controlled_damage": _pairs(damage, "enemy")[:, None]
        * hostile_status[_pair_indices("enemy")[1]],
        "controlled_healing": _pairs(healing, "ally")[:, None]
        * hostile_status[_pair_indices("ally")[1]],
        "controlled_kills": deaths[:, None] & hostile_status,
        "controlled_contributions": _pairs(contributions, "enemy")[:, None]
        & hostile_status[_pair_indices("enemy")[1]],
        "solo_kills": _pairs(
            contributions & (contributor_count == 1)[None, :], "enemy"
        ),
        "multi_kills": _team_sum((deaths & (contributor_count >= 2)).astype(jnp.int32))[
            ::-1
        ],
        "focus_sum": jnp.where(
            focus_eligible, most_shared_target / jnp.maximum(attacking_sources, 1), 0
        ),
        "focus_steps": focus_eligible,
        "trap_periods": new_traps,
        "trap_breaks": trap_breaks,
        "trap_remaining": jnp.where(trap_breaks, state.stun_durations[:, 1] - 1, 0),
        "trap_contributions": _pairs(
            (routed > 0)
            & (combat.raw_damage_output_by_source > 0)[:, None]
            & trap_breaks[None, :],
            "enemy",
        ),
        "waves": facts.respawn_facts.respawn_wave_occurred_this_transition_by_team,
        "respawns": facts.respawn_facts.was_respawned_this_transition_by_agent,
        "ultimate_applications": ultimate_routing,
        "ultimate_damage": damage * ultimate[:, None],
        "ultimate_healing": healing * ultimate[:, None],
        "ultimate_excess": quantities.excess_healing * ultimate[:, None],
        "ultimate_kill_contributions": _pairs(ultimate_contributions, "enemy"),
        "ultimate_kills": ultimate_contributions.any(axis=0),
        "basic_kills": basic_contributions.any(axis=0),
        "class_basic_kills": (
            (basic_contributions[None, :, :] & class_sources[:, :, None])
            .any(axis=1)
            .reshape(5, 2, 5)
            .sum(axis=2)[:, ::-1]
        ),
        "class_ultimate_kills": (
            ultimate_contributions[None, :, :] & class_sources[:, :, None]
        ).any(axis=1),
        "ultimate_rescue_contributions": _pairs(ultimate_rescues, "ally"),
        "ultimate_rescues": ultimate_rescues.any(axis=0),
        "basic_rescues": basic_rescues.any(axis=0),
        "burst_damage": jnp.where(burst[:, None], damage, 0),
        "burst_lethal_damage": _pairs(
            damage * deaths[None, :] * burst[:, None], "enemy"
        ),
        "burst_kills": _pairs(contributions & burst[:, None], "enemy"),
        "burst_unique_kills": (contributions & burst[:, None]).any(axis=0),
        "mage_damage_gain": _team_sum(mage_gain),
        "warrior_damage_prevented": _team_sum(prevented.sum(axis=0)),
        "poison_prevention": poison_prevention.sum(axis=0),
        "rescue_opportunities": quantities.rescue_opportunities,
        "rescues": quantities.rescues,
        "rescue_contributions": _pairs(quantities.rescue_contributions, "ally"),
        "freedom_protected": freedom_protected,
        "freedom_eligible": freedom_eligible,
        "distance_sum": jnp.where(ally_pairs, distances, 0),
        "distance_count": ally_pairs,
    }
    for name, class_id, coverage in (
        (
            "mage",
            1,
            facts.aura_facts.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
        ),
        (
            "warrior",
            2,
            facts.aura_facts.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
        ),
    ):
        covered, eligible, team_covered, team_eligible = _aura_counts(
            config, state, coverage, class_id
        )
        delta[f"{name}_coverage"] = covered
        delta[f"{name}_eligible"] = eligible
        delta[f"{name}_team_coverage"] = team_covered
        delta[f"{name}_team_eligible"] = team_eligible

    return {
        name: total
        if name == "initial_dead"
        else total + jnp.where(facts.has_transition, delta[name], 0).astype(total.dtype)
        for name, total in totals.items()
    }


def _available(value: Array) -> MetricValues:
    return MetricValues(value, jnp.ones(value.shape, bool))


def _ratio(numerator: Array, denominator: Array) -> MetricValues:
    valid = jnp.broadcast_to(denominator > 0, numerator.shape)
    return MetricValues(numerator / jnp.where(valid, denominator, 1), valid)


def full_values(
    totals: FullTotals, config: EnvConfig, priority: MetricValues
) -> MetricValues:
    """Reduce sufficient statistics to the fixed catalog, copying priority values.

    The mappings are static tracing structure, not host dispatch in a rollout.
    Only numerical reductions, ratios and fixed gathers enter compiled execution.
    """
    stored = totals
    expanded = {
        name: _matrix(stored[name], relation)
        for name, relation in (
            ("basic_effective_healing", "ally"),
            ("basic_excess", "ally"),
            ("controlled_damage", "enemy"),
            ("controlled_healing", "ally"),
            ("controlled_contributions", "enemy"),
            ("solo_kills", "enemy"),
            ("trap_contributions", "enemy"),
            ("ultimate_kill_contributions", "enemy"),
            ("ultimate_rescue_contributions", "ally"),
            ("rescue_contributions", "ally"),
            ("burst_kills", "enemy"),
            ("burst_lethal_damage", "enemy"),
            ("mage_coverage", "ally"),
            ("mage_eligible", "ally"),
            ("warrior_coverage", "ally"),
            ("warrior_eligible", "ally"),
        )
    }
    # Preserve the original reductions while retaining directed evidence only
    # once. These local views are not additional episode state.
    totals = {
        **stored,
        **{name: matrix.sum(axis=1) for name, matrix in expanded.items()},
        "activations": jnp.stack(
            (
                stored["basic_applications"].sum(axis=1),
                stored["ultimate_applications"].sum(axis=1),
            ),
            axis=1,
        ),
        **{
            name: _team_sum(stored[name])[::-1]
            for name in (
                "controlled_kills",
                "trap_periods",
                "trap_breaks",
                "trap_remaining",
                "ultimate_kills",
            )
        },
        **{
            name: _team_sum(stored[name])
            for name in (
                "rescues",
                "rescue_opportunities",
                "ultimate_rescues",
                "mage_team_coverage",
                "mage_team_eligible",
                "warrior_team_coverage",
                "warrior_team_eligible",
            )
        },
    }
    data: dict[tuple[MetricScope, str], MetricValues] = {}

    def add(
        name: str,
        *,
        agent: MetricValues | None = None,
        team: MetricValues | None = None,
        pair: MetricValues | None = None,
    ) -> None:
        if agent is not None:
            data[("agent", name)] = agent
        if team is not None:
            data[("team", name)] = team
        if pair is not None:
            data[("source_recipient", name)] = pair
            data[("ally_pair", name)] = pair

    def amount(name: str, agents: Array, teams: Array | None = None) -> None:
        add(
            name,
            agent=_available(agents),
            team=_available(_team_sum(agents) if teams is None else teams),
        )

    def fraction_of_team(name: str, agents: Array) -> None:
        add(name, agent=_ratio(agents, jnp.repeat(_team_sum(agents), 5)))

    deaths = totals["deaths"]
    kills = _team_sum(deaths)[::-1]
    contributions = totals["kills"].sum(axis=1)
    for index, name in enumerate(("basic_activations", "ultimate_activations")):
        amount(name, totals["activations"][:, index])
    add("deaths", agent=_available(deaths))
    fraction_of_team("death_fraction", deaths)
    amount("dead_steps", totals["dead_steps"])
    fraction_of_team("dead_step_fraction", totals["dead_steps"])
    add("kill_contributions", agent=_available(contributions))
    add("kill_contributions_per_death", agent=_ratio(contributions, deaths))
    add("kill_participation", agent=_ratio(contributions, jnp.repeat(kills, 5)))
    for effect in ("damage", "healing"):
        matrix = totals[effect]
        sources = matrix.sum(axis=1)
        amount(f"{effect}_done", sources)
        fraction_of_team(f"{effect}_done_fraction", sources)
        add(f"{effect}_done", pair=_available(matrix))
        team_recipient = jnp.repeat(matrix.reshape(2, 5, 10).sum(axis=1), 5, axis=0)
        add(f"{effect}_done_fraction", pair=_ratio(matrix, team_recipient))
    damage_received = totals["damage"].sum(axis=0)
    priest_received = totals["healing"].sum(axis=0)
    regeneration = totals["regeneration"]
    healing_received = priest_received + regeneration
    amount("damage_received", damage_received)
    fraction_of_team("damage_received_fraction", damage_received)
    amount("priest_healing_received", priest_received)
    amount("regenerated_healing", regeneration)
    amount("healing_received", healing_received)
    for name, component in (
        ("priest", priest_received),
        ("regeneration", regeneration),
    ):
        add(
            f"{name}_healing_received_fraction",
            agent=_ratio(component, healing_received),
            team=_ratio(_team_sum(component), _team_sum(healing_received)),
        )
    fraction_of_team("healing_received_fraction", healing_received)
    amount("excess_healing", totals["excess"].sum(axis=1))
    add("excess_healing_received", agent=_available(totals["excess"].sum(axis=0)))
    # Finalization reuses accumulated source/recipient amounts; no second
    # transition calculation is needed for effective output or efficiency.
    effective = totals["healing"] - totals["excess"]
    effective_sources = effective.sum(axis=1)
    effective_recipients = effective.sum(axis=0)
    delivered_sources = totals["healing"].sum(axis=1)
    excess_sources = totals["excess"].sum(axis=1)
    excess_recipients = totals["excess"].sum(axis=0)
    amount("effective_healing_done", effective_sources)
    amount("effective_priest_healing_received", effective_recipients)
    amount("effective_healing_received", effective_recipients + regeneration)
    for name, numerator, denominator in (
        ("effective_healing_fraction", effective_sources, delivered_sources),
        ("excess_healing_fraction", excess_sources, delivered_sources),
        (
            "effective_priest_healing_received_fraction",
            effective_recipients,
            priest_received,
        ),
        ("excess_priest_healing_received_fraction", excess_recipients, priest_received),
    ):
        add(
            name,
            agent=_ratio(numerator, denominator),
            team=_ratio(_team_sum(numerator), _team_sum(denominator)),
        )
    ultimate_activations = totals["activations"][:, 1]
    add("ultimate_applications", pair=_available(totals["ultimate_applications"]))
    add(
        "ultimate_application_fraction",
        pair=_ratio(totals["ultimate_applications"], ultimate_activations[:, None]),
    )
    for name, matrix in (
        ("ultimate_damage_done", totals["ultimate_damage"]),
        ("ultimate_healing_done", totals["ultimate_healing"]),
        (
            "ultimate_effective_healing_done",
            totals["ultimate_healing"] - totals["ultimate_excess"],
        ),
    ):
        amount(name, matrix.sum(axis=1))
        add(name, pair=_available(matrix))
    ultimate_delivered = totals["ultimate_healing"].sum(axis=1)
    ultimate_excess = totals["ultimate_excess"].sum(axis=1)
    for name, numerator in (
        ("ultimate_effective_healing_fraction", ultimate_delivered - ultimate_excess),
        ("ultimate_excess_healing_fraction", ultimate_excess),
    ):
        add(
            name,
            agent=_ratio(numerator, ultimate_delivered),
            team=_ratio(_team_sum(numerator), _team_sum(ultimate_delivered)),
        )
    for effect, denominator in (("kill", kills), ("rescue", totals["rescues"])):
        agent_count = totals[f"ultimate_{effect}_contributions"]
        team_count = totals[f"ultimate_{effect}s"]
        add(f"ultimate_{effect}_contributions", agent=_available(agent_count))
        add(
            f"ultimate_{effect}_participation",
            agent=_ratio(agent_count, jnp.repeat(denominator, 5)),
        )
        add(f"ultimate_{effect}s", team=_available(team_count))
        add(f"ultimate_{effect}_fraction", team=_ratio(team_count, denominator))
    for channel, status in enumerate(STATUS_NAMES[:7]):
        for effect in ("damage", "healing"):
            amount(
                f"{effect}_to_{status}_recipient",
                totals[f"controlled_{effect}"][:, channel],
            )
        team_kills = totals["controlled_kills"][:, channel]
        agent_kills = totals["controlled_contributions"][:, channel]
        add(f"kills_of_{status}_recipient", team=_available(team_kills))
        add(f"kill_contributions_to_{status}_recipient", agent=_available(agent_kills))
        add(
            f"kill_participation_in_{status}_recipient",
            agent=_ratio(agent_kills, jnp.repeat(team_kills, 5)),
        )
    solo_kills = totals["solo_kills"]
    single_kills = _team_sum(solo_kills)
    add("solo_kills", agent=_available(solo_kills))
    add("solo_kill_fraction", agent=_ratio(solo_kills, contributions))
    add("single_contributor_kills", team=_available(single_kills))
    add("multi_contributor_kills", team=_available(totals["multi_kills"]))
    add("single_contributor_kill_fraction", team=_ratio(single_kills, kills))
    add("multi_contributor_kill_fraction", team=_ratio(totals["multi_kills"], kills))
    add(
        "focus_fire_concentration",
        team=_ratio(totals["focus_sum"], totals["focus_steps"]),
    )
    add("focus_fire_steps", team=_available(totals["focus_steps"]))
    for index, name in enumerate(
        (
            "actions_submitted",
            "actions_accepted",
            "actions_rejected",
            "action_domain_rejections",
            "action_movement_rejections",
            "action_combat_rejections",
        )
    ):
        amount(name, totals["actions"][:, index])
    add(
        "action_acceptance_rate",
        agent=_ratio(totals["actions"][:, 1], totals["actions"][:, 0]),
        team=_ratio(
            _team_sum(totals["actions"][:, 1]), _team_sum(totals["actions"][:, 0])
        ),
    )
    for channel, status in enumerate(STATUS_NAMES):
        class_id = STATUS_CLASS_IDS[channel]
        activation = (1, 0, 1, 1, 1, 1, 1, 1, 0)[channel]
        applications = jnp.where(
            config.agent_profile.active_mask
            & (config.agent_profile.class_ids == class_id),
            totals["activations"][:, activation],
            0,
        )
        amount(f"{status}_applications", applications)
        amount(f"{status}_active_steps", totals["active_steps"][:, channel])
    add(
        "trap_intervals",
        agent=_available(stored["trap_periods"]),
        team=_available(totals["trap_periods"]),
    )
    add("trap_breaks", team=_available(totals["trap_breaks"]))
    add(
        "trap_break_rate",
        agent=_ratio(stored["trap_breaks"], stored["trap_periods"]),
        team=_ratio(totals["trap_breaks"], totals["trap_periods"]),
    )
    add(
        "trap_mean_remaining_steps_at_break",
        agent=_ratio(stored["trap_remaining"], stored["trap_breaks"]),
        team=_ratio(totals["trap_remaining"], totals["trap_breaks"]),
    )
    add("trap_break_contributions", agent=_available(totals["trap_contributions"]))
    add(
        "trap_break_participation",
        agent=_ratio(
            totals["trap_contributions"], jnp.repeat(totals["trap_breaks"], 5)
        ),
    )
    add("respawn_waves", team=_available(totals["waves"]))
    add(
        "mean_agents_per_respawn_wave",
        team=_ratio(_team_sum(totals["respawns"]), totals["waves"]),
    )
    observed_waits = totals["initial_dead"].astype(jnp.int32) + deaths
    add(
        "mean_observed_respawn_wait_steps",
        team=_ratio(_team_sum(totals["dead_steps"]), _team_sum(observed_waits)),
    )
    burst_damage = totals["burst_damage"].sum(axis=1)
    amount("burst_damage", burst_damage)
    add("burst_damage", pair=_available(totals["burst_damage"]))
    amount("burst_damage_contributing_to_kill", totals["burst_lethal_damage"])
    amount("burst_kill_contributions", totals["burst_kills"])
    mage_damage = jnp.where(
        config.agent_profile.class_ids == 1, totals["damage"].sum(axis=1), 0
    )
    add(
        "burst_damage_fraction",
        agent=_ratio(burst_damage, mage_damage),
        team=_ratio(_team_sum(burst_damage), _team_sum(mage_damage)),
    )
    for name in ("mage", "warrior"):
        amount(
            f"{name}_aura_covered_steps",
            totals[f"{name}_coverage"],
            totals[f"{name}_team_coverage"],
        )
        amount(
            f"{name}_aura_eligible_steps",
            totals[f"{name}_eligible"],
            totals[f"{name}_team_eligible"],
        )
        add(
            f"{name}_aura_coverage",
            agent=_ratio(totals[f"{name}_coverage"], totals[f"{name}_eligible"]),
            team=_ratio(
                totals[f"{name}_team_coverage"], totals[f"{name}_team_eligible"]
            ),
        )
    add("damage_from_mage_aura", team=_available(totals["mage_damage_gain"]))
    add(
        "damage_prevented_by_warrior_aura",
        team=_available(totals["warrior_damage_prevented"]),
    )
    amount("healing_prevented_by_poison", totals["poison_prevention"])
    add(
        "rescue_opportunities",
        agent=_available(stored["rescue_opportunities"]),
        team=_available(totals["rescue_opportunities"]),
    )
    add("rescues", team=_available(totals["rescues"]))
    add("rescue_rate", team=_ratio(totals["rescues"], totals["rescue_opportunities"]))
    add("rescue_contributions", agent=_available(totals["rescue_contributions"]))
    add(
        "rescue_participation",
        agent=_ratio(totals["rescue_contributions"], jnp.repeat(totals["rescues"], 5)),
    )
    amount("freedom_protected_steps", totals["freedom_protected"])
    amount("freedom_eligible_steps", totals["freedom_eligible"])
    add(
        "freedom_protection_fraction",
        agent=_ratio(totals["freedom_protected"], totals["freedom_eligible"]),
        team=_ratio(
            _team_sum(totals["freedom_protected"]),
            _team_sum(totals["freedom_eligible"]),
        ),
    )
    distances = totals["distance_sum"]
    distance_count = totals["distance_count"]
    add(
        "ally_distance_mean",
        pair=_ratio(distances, distance_count),
        team=_ratio(
            _team_sum(distances.sum(axis=1)), _team_sum(distance_count.sum(axis=1))
        ),
    )
    add(
        "ally_distance_observations",
        pair=_available(distance_count),
        team=_available(_team_sum(distance_count.sum(axis=1))),
    )

    _directed_values(data, stored, expanded, config)
    return _pack_values(data, config, priority)


def _directed_values(
    data: dict[tuple[MetricScope, str], MetricValues],
    totals: FullTotals,
    expanded: dict[str, Array],
    config: EnvConfig,
) -> None:
    """Expose one shared matrix per family through the catalog's scalar views."""
    matrices = {
        "basic_applications": totals["basic_applications"],
        "ultimate_applications": totals["ultimate_applications"],
        "kill_contributions": totals["kills"],
        "ultimate_kill_contributions": expanded["ultimate_kill_contributions"],
        "basic_kill_contributions": totals["kills"]
        - expanded["ultimate_kill_contributions"],
        "burst_kill_contributions": expanded["burst_kills"],
        "solo_kills": expanded["solo_kills"],
        "rescue_contributions": expanded["rescue_contributions"],
        "ultimate_rescue_contributions": expanded["ultimate_rescue_contributions"],
        "basic_rescue_contributions": expanded["rescue_contributions"]
        - expanded["ultimate_rescue_contributions"],
        "burst_damage": totals["burst_damage"],
        "burst_damage_contributing_to_kill": expanded["burst_lethal_damage"],
        "trap_break_contributions": expanded["trap_contributions"],
    }
    for ability in ("", "basic_", "ultimate_"):
        amounts = {
            effect: totals[effect]
            if not ability
            else totals["ultimate_" + effect]
            if ability == "ultimate_"
            else totals[effect] - totals["ultimate_" + effect]
            for effect in ("damage", "healing")
        }
        if ability == "basic_":
            excess = expanded["basic_excess"]
            effective_healing = expanded["basic_effective_healing"]
        else:
            excess = totals[ability + "excess"]
            effective_healing = amounts["healing"] - excess
        matrices.update(
            {
                ability + "damage_done": amounts["damage"],
                ability + "healing_done": amounts["healing"],
                ability + "effective_healing_done": effective_healing,
                ability + "excess_healing": excess,
            }
        )
        for name, numerator in (
            ("effective_healing_fraction", effective_healing),
            ("excess_healing_fraction", excess),
        ):
            data[("source_recipient", ability + name)] = _ratio(
                numerator, amounts["healing"]
            )
            if ability == "basic_":
                sources = numerator.sum(axis=1)
                delivered = amounts["healing"].sum(axis=1)
                data[("agent", ability + name)] = _ratio(sources, delivered)
                data[("team", ability + name)] = _ratio(
                    _team_sum(sources), _team_sum(delivered)
                )

    # Event denominators retain unique recipient events. In particular a mixed
    # Basic/Ultimate kill or rescue belongs to both ability categories.
    recipients = {
        "deaths": totals["deaths"],
        "rescues": totals["rescues"],
        "trap_breaks": totals["trap_breaks"],
    }
    team_events = {
        "kills": _team_sum(totals["deaths"])[::-1],
        "basic_kills": _team_sum(totals["basic_kills"])[::-1],
        "ultimate_kills": _team_sum(totals["ultimate_kills"])[::-1],
        "burst_kills": _team_sum(totals["burst_unique_kills"])[::-1],
        "single_contributor_kills": _team_sum(expanded["solo_kills"].sum(axis=1)),
        "rescues": _team_sum(totals["rescues"]),
        "basic_rescues": _team_sum(totals["basic_rescues"]),
        "ultimate_rescues": _team_sum(totals["ultimate_rescues"]),
        "trap_breaks": _team_sum(totals["trap_breaks"])[::-1],
    }
    classes = config.agent_profile.class_ids
    for channel, status in enumerate(STATUS_NAMES):
        ability = "basic" if channel in (1, 8) else "ultimate"
        matrices[status + "_applications"] = (
            totals[ability + "_applications"]
            * (classes == STATUS_CLASS_IDS[channel])[:, None]
        )
        if channel < 7:
            for effect in ("damage", "healing"):
                matrices[f"{effect}_to_{status}_recipient"] = expanded[
                    "controlled_" + effect
                ][:, :, channel]
            matrices[f"kill_contributions_to_{status}_recipient"] = expanded[
                "controlled_contributions"
            ][:, :, channel]
            recipients[f"deaths_while_{status}"] = totals["controlled_kills"][
                :, channel
            ]
            team_events[f"kills_of_{status}_recipient"] = _team_sum(
                totals["controlled_kills"][:, channel]
            )[::-1]
    for aura in ("mage", "warrior"):
        for measure, counter in (("covered", "coverage"), ("eligible", "eligible")):
            stem = f"{aura}_aura_{measure}_steps"
            matrices[stem] = expanded[f"{aura}_{counter}"]
            recipients[f"{aura}_aura_{measure}_recipient_steps"] = totals[
                f"{aura}_team_{counter}"
            ]
            team_events[stem] = _team_sum(totals[f"{aura}_team_{counter}"])
        data[("source_recipient", f"{aura}_aura_coverage")] = _ratio(
            expanded[f"{aura}_coverage"], expanded[f"{aura}_eligible"]
        )

    for metric in DIRECTED_METRICS:
        matrix = matrices[metric.key]
        source_key = ("agent", metric.source_total)
        if source_key not in data:
            data[source_key] = _available(matrix.sum(axis=1))
        sources = data[source_key].values
        recipient_amounts = _team_sum(matrix)
        if metric.shared_events:
            recipient_counts = recipients[metric.recipient_total]
            team_counts = team_events[metric.team_total]
        else:
            recipient_counts = matrix.sum(axis=0)
            team_counts = _team_sum(sources)
        data.setdefault(("team", metric.team_total), _available(team_counts))
        if metric.key == "basic_kill_contributions":
            data[("agent", "basic_kill_participation")] = _ratio(
                sources, jnp.repeat(team_events["kills"], 5)
            )
            data[("team", "basic_kill_fraction")] = _ratio(
                team_counts, team_events["kills"]
            )
        elif metric.key == "basic_rescue_contributions":
            data[("agent", "basic_rescue_participation")] = _ratio(
                sources, jnp.repeat(team_events["rescues"], 5)
            )
            data[("team", "basic_rescue_fraction")] = _ratio(
                team_counts, team_events["rescues"]
            )
        data.setdefault(
            (metric.recipient_scope, metric.recipient_total),
            _available(
                recipient_amounts
                if metric.recipient_scope == "team_recipient"
                else recipient_counts
            ),
        )
        data[("source_recipient", metric.amount)] = _available(matrix)
        if metric.allocation is not None:
            data.setdefault(
                ("source_recipient", metric.allocation),
                _ratio(matrix, sources[:, None]),
            )
        if metric.contribution is not None:
            denominator = (
                recipient_counts[None, :]
                if metric.shared_events
                else jnp.repeat(recipient_amounts, 5, axis=0)
            )
            data.setdefault(
                ("source_recipient", metric.contribution), _ratio(matrix, denominator)
            )
    for class_id, class_name in enumerate(ULTIMATE_CLASS_NAMES, 1):
        kills = totals["class_basic_kills"][class_id - 1]
        data[("team", class_name + "_basic_kills")] = _available(kills)
        data[("team", class_name + "_basic_kill_fraction")] = _ratio(
            kills, team_events["kills"]
        )
    for class_id, class_name in enumerate(ULTIMATE_CLASS_NAMES[1:], 2):
        kills = _team_sum(totals["class_ultimate_kills"][class_id - 1])[::-1]
        data[("team", class_name + "_ultimate_kills")] = _available(kills)
        data[("team", class_name + "_ultimate_kill_fraction")] = _ratio(
            kills, team_events["kills"]
        )


@cache
def _packing_groups() -> tuple[tuple[tuple[MetricScope, str], tuple[int, ...]], ...]:
    """Group static column indices so tracing never creates one kernel per cell."""
    groups: dict[tuple[MetricScope, str], list[int]] = {}
    for column in METRIC_COLUMNS[len(PRIORITY_METRIC_COLUMNS) :]:
        if column.scope == "team":
            index = column.subjects[0] - 1
        elif column.scope == "team_recipient":
            index = (column.subjects[0] - 1) * 10 + column.subjects[1]
        elif column.scope in ("source_recipient", "ally_pair"):
            index = column.subjects[0] * 10 + column.subjects[1]
        else:
            index = column.subjects[0]
        groups.setdefault((column.scope, column.stem), []).append(index)
    return tuple((key, tuple(indices)) for key, indices in groups.items())


@cache
def _packing_order() -> tuple[int, ...]:
    groups: dict[tuple[MetricScope, str], list[int]] = {}
    for index, column in enumerate(METRIC_COLUMNS[len(PRIORITY_METRIC_COLUMNS) :]):
        groups.setdefault((column.scope, column.stem), []).append(index)
    grouped = [index for indices in groups.values() for index in indices]
    return tuple(sorted(range(len(grouped)), key=grouped.__getitem__))


def _pack_values(
    data: dict[tuple[MetricScope, str], MetricValues],
    config: EnvConfig,
    priority: MetricValues,
) -> MetricValues:
    values: list[Array] = []
    valid: list[Array] = []
    for key, indices in _packing_groups():
        measurement = data[key]
        indices_array = jnp.asarray(indices)
        values.append(measurement.values.reshape(-1)[indices_array])
        valid.append(measurement.valid.reshape(-1)[indices_array])
    order = jnp.asarray(_packing_order())
    result = jnp.concatenate(values)[order].astype(jnp.float32)
    available = jnp.concatenate(valid)[order]
    columns = METRIC_COLUMNS[len(PRIORITY_METRIC_COLUMNS) :]
    team_scope = jnp.asarray(
        tuple(c.scope in ("team", "team_recipient") for c in columns)
    )
    pair_scope = jnp.asarray(
        tuple(
            c.scope in ("source_recipient", "ally_pair", "team_recipient")
            for c in columns
        )
    )
    source = jnp.asarray(
        tuple(
            0 if c.scope in ("team", "team_recipient") else c.subjects[0]
            for c in columns
        )
    )
    recipient = jnp.asarray(
        tuple(c.subjects[1] if len(c.subjects) == 2 else 0 for c in columns)
    )
    team = jnp.asarray(
        tuple(
            c.subjects[0] - 1
            if c.scope in ("team", "team_recipient")
            else c.subjects[0] // 5
            for c in columns
        )
    )
    required_class = jnp.asarray(tuple(c.required_class_id or 0 for c in columns))
    active = config.agent_profile.active_mask
    classes = config.agent_profile.class_ids
    class_sources = (jnp.arange(6)[:, None] == classes[None, :]) | (
        jnp.arange(6)[:, None] == 0
    )
    class_sources = class_sources & active[None, :]
    team_classes = class_sources.reshape(6, 2, 5).any(axis=2)
    available &= jnp.where(
        team_scope, active.reshape(2, 5).any(axis=1)[team], active[source]
    )
    available &= ~pair_scope | active[recipient]
    available &= jnp.where(
        team_scope,
        team_classes[required_class, team],
        (required_class == 0) | (classes[source] == required_class),
    )
    recipient_rescue_opportunity = jnp.asarray(
        tuple(c.scope == "agent" and c.stem == "rescue_opportunities" for c in columns)
    )
    available &= ~recipient_rescue_opportunity | team_classes[5, team]
    same_team = (
        config.agent_profile.team_ids[:, None] == config.agent_profile.team_ids[None, :]
    )
    modes = get_ultimate_target_mode_by_class_ids(classes)[:, None]
    ultimate_targets = (
        ((modes == ONLY_NONE_TARGET_ULTIMATE_MODE) & jnp.eye(10, dtype=bool))
        | ((modes == ONLY_ALLY_TARGET_ULTIMATE_MODE) & same_team)
        | ((modes == ONLY_ENEMY_TARGET_ULTIMATE_MODE) & ~same_team)
    )
    basic_targets = jnp.where((classes == 5)[:, None], same_team, ~same_team)
    for targets, flags in (
        (basic_targets, tuple(c.requires_basic_target for c in columns)),
        (ultimate_targets, tuple(c.requires_ultimate_target for c in columns)),
    ):
        team_targets = (
            (class_sources[:, :, None] & targets[None, :, :])
            .reshape(6, 2, 5, 10)
            .any(axis=2)
        )
        target_available = jnp.where(
            team_scope,
            team_targets[required_class, team, recipient],
            targets[source, recipient],
        )
        available &= ~(jnp.asarray(flags) & pair_scope) | target_available
    return MetricValues(
        jnp.concatenate((priority.values, result)),
        jnp.concatenate((priority.valid, available)),
    )
