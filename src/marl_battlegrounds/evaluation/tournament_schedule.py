"""Build deterministic all-pairs schedules with equal map and spawn-bank exposure.

One block holds two games with fixed Team A/B ownership and opposite complete
spawn banks. They share a source map and seed. Schedule creation performs no simulation
or random draw. Statistics later treat each block, or an explicitly larger
coupled group, as one resampling unit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral


def valid_policy_name(value: object) -> bool:
    """Return whether a value is a nonblank policy-name string.

    Parameters
    ----------
    value : object
        Candidate label. Leading/trailing whitespace is allowed but not removed.

    Returns
    -------
    bool
        True only for a string containing at least one non-whitespace character.
    """
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class TournamentMatch:
    """One scheduled game and its paired random-stream identity.

    Attributes
    ----------
    episode_id : int
        Unique positive int32 row identity.
    block_id : int
        Positive identity joining the two complementary games.
    seed_id : int
        uint32 stream coordinate folded into the tournament root seed.
    map_id : int
        Nonnegative integer map ID used by both games in the block.
    team_a : str
        Policy name assigned to Team A for this game.
    team_b : str
        Different policy name assigned to Team B.
    bootstrap_group : str | None
        Optional nonempty name declaring dependence across several
        blocks. None treats this block as its own independent unit.

    pairing_protocol : str | None
        "fixed-team-spawn-v1" for new schedules. None or "team-swap-v1"
        preserves historical reversed-team records.
    spawn_locations : int | None
        Source bank order (0) or complete exchange (1); absent for old records.
    source_config_id, resolved_config_id : str | None
        Configuration content references filled by the runner. IDs alone do not
        prove that actual configurations form a physical spawn pair.

    Notes
    -----
    This frozen description stores values; the schedule/statistics helpers
    validate them. Sharing a seed does not guarantee equal actions under
    different policy inputs. Statistical validation checks logical pair structure;
    the runner separately verifies physical source and resolved configurations.
    """

    episode_id: int
    block_id: int
    seed_id: int
    map_id: int
    team_a: str
    team_b: str
    bootstrap_group: str | None = None
    pairing_protocol: str | None = None
    spawn_locations: int | None = None
    source_config_id: str | None = None
    resolved_config_id: str | None = None


def build_tournament_schedule(
    policies: Sequence[str],
    maps: Sequence[int],
    *,
    episodes_per_pair: int = 100,
) -> tuple[TournamentMatch, ...]:
    """Give each unordered pair equal maps and opposite spawn-bank choices.

    Parameters
    ----------
    policies : Sequence[str]
        At least two distinct nonblank policy-name strings.
    maps : Sequence[int]
        Nonempty distinct nonnegative int32-compatible map IDs. This helper
        checks ID form, not whether packaged geometry exists.
    episodes_per_pair : int
        Positive total across maps and both spawn choices, default 100.
        Must be divisible by twice the number of maps.

    Returns
    -------
    tuple[TournamentMatch, ...]
        Tuple of TournamentMatch in sorted pair, sorted map, block and side order.
        IDs are sequential from 1. The sorted first participant stays Team A in
        both games. Each block uses choices 0 and 1 and seed_id equal to block_id.
        bootstrap_group defaults to None.

    Raises
    ------
    ValueError
        Names/maps/budget are invalid, or total episode IDs exceed int32.

    Notes
    -----
    Host-only and deterministic. Sorting removes dependence on input ordering;
    no RNG or files are used. This is the current all-pairs schedule, not a
    future one-entrant canonical tournament protocol.
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
                for spawn_locations in (0, 1):
                    matches.append(
                        TournamentMatch(
                            episode_id=len(matches) + 1,
                            block_id=block_id,
                            seed_id=block_id,
                            map_id=map_id,
                            team_a=first,
                            team_b=second,
                            pairing_protocol="fixed-team-spawn-v1",
                            spawn_locations=spawn_locations,
                        )
                    )
    return tuple(matches)
