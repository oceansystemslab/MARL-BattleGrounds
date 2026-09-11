"""Combat measurements against accepted public Core transition witnesses."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.match_summary import build_match_summary_v1
from tests.evaluation_fixtures import (
    evaluation_context,
    evaluation_env_config,
    neutral_action,
)

from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.combat_metrics import (
    CombatQuantities,
    combat_quantities,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2

_step = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_quantities = cast(
    Callable[[EnvConfig, EnvState, ActionMask, Info], CombatQuantities],
    jax.jit(combat_quantities),
)


def _start(
    *,
    class_changes: tuple[tuple[int, int], ...] = (),
    team_sizes: tuple[int, int] = (5, 5),
    arrange: Callable[[EnvState], EnvState] | None = None,
) -> tuple[EnvConfig, EnvState, ActionMask]:
    config = evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=1,
        team_deathmatch_score_threshold=20,
        max_steps=4,
    )
    classes = config.agent_profile.class_ids
    for slot, class_id in class_changes:
        classes = classes.at[slot].set(class_id)
    config = config._replace(
        agent_profile=resolve_agent_profile(classes, jnp.asarray(team_sizes, jnp.int32))
    )
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = (
        jnp.asarray(
            [(6.0, 1.5 + 2 * slot) for slot in range(5)]
            + [(9.0, 1.5 + 2 * slot) for slot in range(5)],
            dtype=jnp.float32,
        )
        .at[2]
        .set((6.0, 2.5))
    )
    state = state._replace(
        agent_positions=jnp.where(
            config.agent_profile.active_mask[:, None], positions, 0
        ),
        spawn_shield_durations=jnp.zeros(10, jnp.int32),
    )
    if arrange is not None:
        state = arrange(state)
    state, _, mask, _ = initialize_scenario_state(state, config)
    return config, state, mask


def _actions(*choices: tuple[int, int, bool]) -> Action:
    action = neutral_action()
    for source, recipient, ultimate in choices:
        action = action._replace(
            select_target=action.select_target.at[source].set(
                global_slot_to_target_action(source, recipient)
            ),
            use_ultimate=action.use_ultimate.at[source].set(int(ultimate)),
        )
    return action


def _advance(
    case: tuple[EnvConfig, EnvState, ActionMask], action: Action
) -> tuple[EnvState, Info, CombatQuantities]:
    config, state, mask = case
    successor, _, _, _, _, info = _step(config, state, mask, action, jax.random.key(1))
    return successor, info, _quantities(config, state, mask, info)


def test_delivered_matrices_keep_overkill_and_separate_done_from_received() -> None:
    case = _start(
        arrange=lambda state: state._replace(
            current_health=state.current_health.at[0].set(40).at[5].set(1),
            rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                0
            ].set(2),
        )
    )
    successor, info, values = _advance(case, _actions((0, 5, False), (2, 0, False)))
    facts = info.transition_facts.combat_transition_facts
    np.testing.assert_allclose(
        values.damage.sum(axis=0),
        facts.total_effective_damage_by_recipient,
        rtol=1e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        values.healing.sum(axis=0),
        facts.total_effective_healing_by_recipient,
        rtol=1e-6,
        atol=1e-5,
    )
    assert values.damage[0, 5] > 1  # Retain damage beyond the victim's remaining HP.
    assert not successor.alive_mask[5]
    assert values.damage[2].sum() == 0  # Priest did no damage.
    assert values.healing[0].sum() == 0  # Mage did no healing.
    assert values.healing[2, 0] == 4  # Poison modifies the delivered Basic heal.
    assert values.healing[:, 0].sum() > 0  # The Mage received the Priest's healing.
    assert values.kill_contributions[0, 5]
    assert values.kill_contributions[2, 5]


@pytest.mark.parametrize("missing_health", (0.0, 0.5, 16.0))
def test_proportional_overheal_only_credits_useful_priest_support(
    missing_health: float,
) -> None:
    case = _start(
        class_changes=((3, 5),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[3].set((7.0, 2.5)),
            current_health=state.current_health.at[0]
            .set(80 - missing_health)
            .at[5]
            .set(1),
        ),
    )
    _, _, values = _advance(case, _actions((0, 5, False), (2, 0, False), (3, 0, False)))
    np.testing.assert_allclose(values.healing[[2, 3], 0], (8, 8))
    np.testing.assert_allclose(
        values.excess_healing[[2, 3], 0], ((16 - missing_health) / 2,) * 2
    )
    assert bool(values.kill_contributions[2, 5]) == (missing_health > 0)
    assert bool(values.kill_contributions[3, 5]) == (missing_health > 0)
    assert values.kill_contributions[0, 5]
    assert values.kill_contributions.sum() == (3 if missing_health else 1)


def test_priest_support_does_not_propagate_recursively() -> None:
    case = _start(
        class_changes=((3, 5),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[3].set((7.0, 2.5)),
            current_health=state.current_health.at[0]
            .set(40)
            .at[2]
            .set(50)
            .at[5]
            .set(1),
        ),
    )
    _, _, values = _advance(case, _actions((0, 5, False), (2, 0, False), (3, 2, False)))
    assert values.healing[3, 2] > values.excess_healing[3, 2]
    assert values.kill_contributions[0, 5]
    assert values.kill_contributions[2, 5]
    assert not values.kill_contributions[3, 5]


def test_dying_priest_keeps_same_tick_accepted_useful_healing_credit() -> None:
    case = _start(
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[6].set((7.0, 2.5)),
            current_health=state.current_health.at[0].set(40).at[2].set(1).at[5].set(1),
        )
    )
    successor, info, values = _advance(
        case, _actions((0, 5, False), (2, 0, False), (6, 2, False))
    )
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient[2]
    assert not successor.alive_mask[2]
    assert values.healing[2, 0] == 8
    assert values.kill_contributions[2, 5]


def test_direct_and_priest_contributors_share_one_victim_without_extra_kills() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[1]
            .set((7.0, 3.0))
            .at[2]
            .set((5.0, 2.5))
            .at[3]
            .set((5.0, 4.0))
            .at[5]
            .set((8.5, 1.5)),
            current_health=state.current_health.at[0]
            .set(40)
            .at[1]
            .set(40)
            .at[5]
            .set(1),
        )

    case = _start(class_changes=((1, 1), (3, 5)), arrange=arrange)
    _, info, values = _advance(
        case, _actions((0, 5, False), (1, 5, False), (2, 0, False), (3, 1, False))
    )
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient.sum() == 1
    np.testing.assert_array_equal(values.kill_contributions[:4, 5], True)
    assert values.kill_contributions.sum() == 4


@pytest.mark.parametrize("missing_health", (0.0, 0.5, 16.0))
@pytest.mark.parametrize("dying_priest", (False, True))
def test_death_announcer_matches_public_direct_and_priest_credit(
    missing_health: float, dying_priest: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded global events preserve support, excess and simultaneous deaths."""
    config, state, mask = _start(
        class_changes=((1, 1), (3, 5)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[1]
            .set((8.0, 3.5))
            .at[3]
            .set((5.0, 2.5))
            .at[6]
            .set((7.0, 2.5)),
            current_health=state.current_health.at[0]
            .set(80 - missing_health)
            .at[2]
            .set(1 if dying_priest else 50)
            .at[5]
            .set(1),
        ),
    )
    state, observation, mask, _ = initialize_scenario_state(state, config)
    context = evaluation_context(config=config, expected_horizon=4, with_scenario=True)
    start = capture_initial_evaluation_frame_v1(context, state, observation, mask)
    action = _actions(
        (0, 5, False),
        (1, 5, False),
        (2, 0, False),
        (3, 0, False),
        *((6, 2, False),) if dying_priest else (),
    )
    successor, observation, reward, done, next_mask, info = _step(
        config, state, mask, action, jax.random.key(1)
    )
    values = _quantities(config, state, mask, info)
    transition, end = capture_evaluation_transition_unit_v1(
        context,
        start,
        successor,
        observation,
        next_mask,
        info.transition_facts,
        reward,
        done,
    )
    view = EvaluationTransitionViewV1(
        context=context, start_frame=start, transition=transition, successor_frame=end
    )
    expected = (0, 1, 2, 3) if missing_health else (0, 1)
    np.testing.assert_array_equal(
        np.flatnonzero(values.kill_contributions[:, 5]), expected
    )
    assert (
        bool(info.transition_facts.death_facts.is_newly_dead_by_recipient[2])
        == dying_priest
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "the HUD must not compute full metrics or rescue counterfactuals"
        )

    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.combat_metrics._available_priest_healing",
        forbidden,
    )
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.combat_metrics.combat_quantities", forbidden
    )
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.full_metrics.update_full", forbidden
    )
    for events in (transition.events, build_visual_event_batch_v2(view).events):
        summary = build_match_summary_v1(context, end, events)
        victims = {row.public_agent_id: row for row in summary.deaths}
        for victim in np.flatnonzero(
            info.transition_facts.death_facts.is_newly_dead_by_recipient
        ):
            death = victims[context.roster[victim].public_agent_id]
            assert (
                death.killing_team_id == 3 - context.roster[victim].configured_team_id
            )
            assert death.contributors is not None
            assert tuple(row.public_agent_id for row in death.contributors) == tuple(
                context.roster[source].public_agent_id
                for source in np.flatnonzero(values.kill_contributions[:, victim])
            )
        # No successor-alive filter may remove a dying Priest's accepted support.
        assert tuple(
            row.public_agent_id
            for row in victims[context.roster[5].public_agent_id].contributors or ()
        ) == tuple(context.roster[source].public_agent_id for source in expected)
    # A historical subset may omit support evidence. Known direct attackers
    # alone must not masquerade as a complete list of Kill Contributors.
    for omitted in ("source_healing_output", "recipient_health_resolution"):
        incomplete = tuple(
            event for event in transition.events if event.event_type != omitted
        )
        sparse = build_match_summary_v1(context, end, incomplete)
        assert all(
            row.killing_team_id is None and row.contributors is None
            for row in sparse.deaths
        )
    credit_event = next(
        event
        for event in transition.events
        if event.event_type == "lethal_damage_contribution"
    )
    incomplete_direct = tuple(
        event for event in transition.events if event is not credit_event
    )
    incomplete_support = tuple(
        event
        for event in transition.events
        if not (
            event.event_type == "source_healing_output"
            or (
                event.event_type == "recipient_health_resolution"
                and event.recipient_global_slot == 0
            )
        )
    )
    last_credit = next(
        event
        for event in reversed(transition.events)
        if event.event_type == "lethal_damage_contribution"
    )
    incomplete_tail = tuple(
        event for event in transition.events if event.ordinal < last_credit.ordinal
    )
    assert tuple(event.ordinal for event in incomplete_tail) == tuple(
        range(len(incomplete_tail))
    )
    for incomplete in (incomplete_direct, incomplete_support, incomplete_tail):
        sparse = build_match_summary_v1(context, end, incomplete)
        assert sparse.deaths
        assert all(
            row.contributors is None and row.killing_team_id is None
            for row in sparse.deaths
        )
    with pytest.raises(ValueError, match="incoming transition"):
        build_match_summary_v1(
            context,
            end,
            tuple(
                event.model_copy(
                    update={"transition_id": f"{end.episode_id}:transition:99"}
                )
                if event is credit_event
                else event
                for event in transition.events
            ),
        )


def _combined_rescue_start(
    block: str | None = None,
) -> tuple[EnvConfig, EnvState, ActionMask]:
    def arrange(state: EnvState) -> EnvState:
        state = state._replace(
            agent_positions=state.agent_positions.at[3].set((7.0, 2.5)),
            current_health=state.current_health.at[0].set(1),
            ultimate_cooldowns=state.ultimate_cooldowns.at[2].set(5).at[3].set(5),
        )
        if block == "stun":
            state = state._replace(stun_durations=state.stun_durations.at[3, 0].set(1))
        elif block == "shield":
            state = state._replace(
                spawn_shield_durations=state.spawn_shield_durations.at[3].set(1)
            )
        elif block == "range":
            state = state._replace(
                agent_positions=state.agent_positions.at[3].set((1.5, 9.5))
            )
        elif block == "poison":
            state = state._replace(
                rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                    0
                ].set(2)
            )
        return state

    return _start(class_changes=((3, 5), (5, 1)), arrange=arrange)


def test_rescue_opportunity_requires_enough_combined_legal_healing() -> None:
    case = _combined_rescue_start()
    for healers in ((), ((2, 0, False),), ((2, 0, False), (3, 0, False))):
        successor, _, values = _advance(case, _actions((5, 0, False), *healers))
        assert values.rescue_opportunities[0]
        assert bool(values.rescues[0]) == (len(healers) == 2)
        assert bool(successor.alive_mask[0]) == (len(healers) == 2)
        assert values.rescue_contributions.sum() == (2 if len(healers) == 2 else 0)


@pytest.mark.parametrize("block", ("stun", "shield", "range", "poison"))
def test_legal_or_effective_healing_limits_remove_rescue_opportunity(
    block: str,
) -> None:
    case = _combined_rescue_start(block)
    successor, _, values = _advance(
        case, _actions((5, 0, False), (2, 0, False), (3, 0, False))
    )
    assert not values.rescue_opportunities[0]
    assert not values.rescues[0]
    assert not successor.alive_mask[0]
    if block in ("stun", "shield", "range"):
        assert values.healing[3].sum() == 0


@pytest.mark.parametrize("cooldown", (0, 1))
def test_available_ultimate_uses_actual_mask_and_real_heal_capacity(
    cooldown: int,
) -> None:
    case = _start(
        class_changes=((5, 1),),
        arrange=lambda state: state._replace(
            current_health=state.current_health.at[0].set(1),
            ultimate_cooldowns=state.ultimate_cooldowns.at[2].set(cooldown),
        ),
    )
    _, _, idle = _advance(case, _actions((5, 0, False)))
    successor, _, attempted = _advance(case, _actions((5, 0, False), (2, 0, True)))
    assert bool(idle.rescue_opportunities[0]) == (cooldown == 0)
    assert not idle.rescues[0]
    assert bool(attempted.rescues[0]) == (cooldown == 0)
    assert bool(successor.alive_mask[0]) == (cooldown == 0)


def test_health_exactly_zero_is_not_a_rescue_or_possible_basic_rescue() -> None:
    case = _start(
        class_changes=((5, 4),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[1]
            .set((1.5, 10.5))
            .at[5]
            .set((7.5, 1.5)),
            current_health=state.current_health.at[0].set(4),
            ultimate_cooldowns=state.ultimate_cooldowns.at[2].set(5),
        ),
    )
    successor, info, values = _advance(case, _actions((5, 0, False), (2, 0, False)))
    assert values.healing[2, 0] == 8
    assert values.damage[5, 0] == 12
    assert (
        info.transition_facts.combat_transition_facts.health_after_combat_resolution_by_recipient[
            0
        ]
        == 0
    )
    assert not successor.alive_mask[0]
    assert not values.rescue_opportunities[0]
    assert not values.rescues[0]


def test_regeneration_and_padded_slots_never_become_healing_or_rescue_credit() -> None:
    case = _start(
        team_sizes=(3, 2),
        arrange=lambda state: state._replace(
            current_health=state.current_health.at[0].set(40)
        ),
    )
    _, info, values = _advance(case, neutral_action())
    assert (
        info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            0
        ]
        > 0
    )
    for array in values:
        np.testing.assert_array_equal(array, 0)
        assert np.isfinite(array).all()


def test_public_witness_has_identical_eager_jit_and_batched_quantities() -> None:
    case = _combined_rescue_start()
    config, state, mask = case
    _, info, compiled = _advance(
        case, _actions((5, 0, False), (2, 0, False), (3, 0, False))
    )
    eager = combat_quantities(config, state, mask, info)

    def duplicate(value: Array) -> Array:
        return jnp.stack((value, value))

    batched = cast(
        CombatQuantities,
        jax.jit(jax.vmap(combat_quantities, in_axes=(None, None, None, 0)))(
            config, state, mask, jax.tree.map(duplicate, info)
        ),
    )
    for expected, actual, pair in zip(eager, compiled, batched, strict=True):
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(pair[0], actual)
        np.testing.assert_array_equal(pair[1], actual)
        assert actual.shape in ((10,), (10, 10))
        assert actual.dtype in (jnp.float32, jnp.bool_)
