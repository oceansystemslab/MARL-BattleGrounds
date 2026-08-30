"""Lazy compatibility exports for canonical fixed-slot target conversion.

Importing this host-side module must not initialize the simulator array stack.
The canonical core implementation remains the sole runtime authority and is
loaded only when a conversion is actually requested.
"""


def global_slot_to_target_action(
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Delegate one fixed-slot conversion to the canonical core mapping."""
    from marl_battlegrounds.core.axis_mappings import (
        global_slot_to_target_action as _global_slot_to_target_action,
    )

    return _global_slot_to_target_action(actor_global_slot, target_global_slot)


def target_action_to_global_slot(
    actor_global_slot: int,
    target_action: int,
) -> int | None:
    """Delegate one target-category conversion to the canonical core mapping."""
    from marl_battlegrounds.core.axis_mappings import (
        target_action_to_global_slot as _target_action_to_global_slot,
    )

    return _target_action_to_global_slot(actor_global_slot, target_action)


__all__ = ["global_slot_to_target_action", "target_action_to_global_slot"]
