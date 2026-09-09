"""Deterministic map-balanced schedules with both sides in each seed block."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral


def valid_policy_name(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class TournamentMatch:
    """One episode and its independent resampling/random-stream coordinate.

    The executor folds ``seed_id`` into its root seed. Both side assignments
    share that coordinate. ``bootstrap_group`` declares additional dependence
    across blocks, when the experimental design deliberately introduces it.
    """

    episode_id: int
    block_id: int
    seed_id: int
    map_id: int
    team_a: str
    team_b: str
    bootstrap_group: str | None = None


def build_tournament_schedule(
    policies: Sequence[str],
    maps: Sequence[int],
    *,
    episodes_per_pair: int = 100,
) -> tuple[TournamentMatch, ...]:
    """Allocate equal maps and opposite sides for every unordered policy pair.

    Names and maps are sorted before assigning identities, so input ordering,
    execution batch size and resume order cannot alter an episode's seed.
    ``episodes_per_pair`` must be divisible by twice the number of maps.
    """
    if isinstance(policies, str) or not all(map(valid_policy_name, policies)):
        raise ValueError("Policies must have nonempty string identities")
    names = tuple(sorted(policies))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("A tournament needs at least two distinct policies")
    if any(
        isinstance(map_id, bool)
        or not isinstance(map_id, Integral)
        or not 0 <= map_id <= 2**31 - 1
        for map_id in maps
    ):
        raise ValueError("Map IDs must be nonnegative integers")
    map_ids = tuple(sorted(int(map_id) for map_id in maps))
    if not map_ids or len(set(map_ids)) != len(map_ids):
        raise ValueError("A tournament needs distinct, nonempty map IDs")
    if (
        isinstance(episodes_per_pair, bool)
        or not isinstance(episodes_per_pair, Integral)
        or episodes_per_pair <= 0
        or episodes_per_pair % (2 * len(map_ids))
    ):
        raise ValueError("episodes_per_pair must be divisible by twice the map count")
    match_count = len(names) * (len(names) - 1) // 2 * episodes_per_pair
    if match_count > 2**31 - 1:
        raise ValueError("Tournament episode IDs exceed the positive int32 domain")

    matches: list[TournamentMatch] = []
    block_id = 0
    for first, second in combinations(names, 2):
        for map_id in map_ids:
            for _ in range(episodes_per_pair // (2 * len(map_ids))):
                block_id += 1
                for team_a, team_b in ((first, second), (second, first)):
                    matches.append(
                        TournamentMatch(
                            episode_id=len(matches) + 1,
                            block_id=block_id,
                            seed_id=block_id,
                            map_id=map_id,
                            team_a=team_a,
                            team_b=team_b,
                        )
                    )
    return tuple(matches)
