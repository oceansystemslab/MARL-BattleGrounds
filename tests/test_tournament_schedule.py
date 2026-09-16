"""Check tournament budgets, spawn pairs and random-stream identities.

These schedule tests do not need to play the scheduled games.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import cast

import pytest

from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    build_tournament_schedule,
)
from marl_battlegrounds.tasks import CANONICAL_TDM_EVALUATION_MAP_IDS


def test_big_twelve_has_exact_pair_map_side_and_seed_balance() -> None:
    names = tuple(f"method-{i:02}" for i in range(12))
    schedule = build_tournament_schedule(names, CANONICAL_TDM_EVALUATION_MAP_IDS)
    assert len(schedule) == 6600
    assert [match.episode_id for match in schedule] == list(range(1, 6601))
    pairs = Counter(tuple(sorted((m.team_a, m.team_b))) for m in schedule)
    assert pairs == {pair: 100 for pair in combinations(names, 2)}
    assert Counter((m.team_a, m.team_b, m.map_id) for m in schedule) == {
        (a, b, map_id): 10
        for a in names
        for b in names
        if a != b
        for map_id in CANONICAL_TDM_EVALUATION_MAP_IDS
    }
    blocks: dict[int, list[TournamentMatch]] = defaultdict(list)
    for match in schedule:
        blocks[match.block_id].append(match)
    assert len(blocks) == len({m.seed_id for m in schedule}) == 3300
    for first, second in blocks.values():
        assert (first.team_a, first.team_b) == (second.team_b, second.team_a)
        assert first.map_id == second.map_id
        assert first.seed_id == second.seed_id == first.block_id == second.block_id


@pytest.mark.parametrize("entrants", (2, 3, 13))
def test_custom_schedules_preserve_input_order_independence(entrants: int) -> None:
    names = tuple(f"policy-{i:02}" for i in range(entrants))
    maps = (9, 2, 4)
    schedule = build_tournament_schedule(names, maps, episodes_per_pair=12)
    assert len(schedule) == entrants * (entrants - 1) // 2 * 12
    assert schedule == build_tournament_schedule(
        names[::-1], maps[::-1], episodes_per_pair=12
    )
    # Reusing the coordinate across maps or unrelated opponents would silently
    # couple nominally independent evaluation evidence.
    identities = {
        (m.block_id, m.map_id, tuple(sorted((m.team_a, m.team_b)))) for m in schedule
    }
    assert len(identities) == len({m.seed_id for m in schedule})


@pytest.mark.parametrize(
    ("names", "maps", "budget"),
    [
        (("only",), (1,), 2),
        (("a", "a"), (1,), 2),
        (("a", " "), (1,), 2),
        (("a", 2), (1,), 2),
        ("ab", (1,), 2),
        (("a", "b"), (), 2),
        (("a", "b"), (1, 1), 4),
        (("a", "b"), (-1,), 2),
        (("a", "b"), (2**31,), 2),
        (("a", "b"), (True,), 2),
        (("a", "b"), (1,), True),
        (("a", "b"), (1,), 2.5),
        (("a", "b"), (1,), 0),
        (("a", "b"), (1, 2), 6),
        (("a", "b"), (1,), 2**31),
    ],
)
def test_invalid_schedule_inputs_fail_before_allocation(
    names: object, maps: object, budget: object
) -> None:
    with pytest.raises(ValueError):
        build_tournament_schedule(
            cast(Sequence[str], names),
            cast(Sequence[int], maps),
            episodes_per_pair=cast(int, budget),
        )
