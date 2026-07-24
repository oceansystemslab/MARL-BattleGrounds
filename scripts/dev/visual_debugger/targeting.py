"""Debugger-local fixed-slot and actor-relative target conversion."""

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_TARGET_ACTIONS,
)


def _validate_actor(actor_global_slot: int) -> None:
    if not 0 <= actor_global_slot < MAX_AGENT_SLOTS:
        msg = (
            "actor_global_slot must be a fixed global slot in "
            f"[0, {MAX_AGENT_SLOTS}); got {actor_global_slot}."
        )
        raise ValueError(msg)


def global_slot_to_target_action(
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Convert a clicked global slot into the actor-relative target category."""
    _validate_actor(actor_global_slot)
    if target_global_slot is None:
        return 0
    if not 0 <= target_global_slot < MAX_AGENT_SLOTS:
        msg = (
            "target_global_slot must be None or a fixed global slot in "
            f"[0, {MAX_AGENT_SLOTS}); got {target_global_slot}."
        )
        raise ValueError(msg)

    if actor_global_slot < MAX_AGENTS_PER_TEAM:
        return target_global_slot + 1
    if target_global_slot >= MAX_AGENTS_PER_TEAM:
        return target_global_slot - MAX_AGENTS_PER_TEAM + 1
    return target_global_slot + MAX_AGENTS_PER_TEAM + 1


def target_action_to_global_slot(
    actor_global_slot: int,
    target_action: int,
) -> int | None:
    """Invert the fixed actor-relative target category mapping."""
    _validate_actor(actor_global_slot)
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        msg = (
            f"target_action must be in [0, {NUM_TARGET_ACTIONS}); got {target_action}."
        )
        raise ValueError(msg)
    if target_action == 0:
        return None

    relation_row = target_action - 1
    if actor_global_slot < MAX_AGENTS_PER_TEAM:
        return relation_row
    if relation_row < MAX_AGENTS_PER_TEAM:
        return relation_row + MAX_AGENTS_PER_TEAM
    return relation_row - MAX_AGENTS_PER_TEAM
