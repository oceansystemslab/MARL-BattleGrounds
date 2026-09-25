"""Name legal actor choices and check decoded replies before Core submission.

Core owns native categories and display names. This module owns their stable
machine spelling and the default two-key JSON reply. Custom parsers reuse
validate_actor_action. These host-only helpers accept one actor in one frame;
they neither reflect movement nor repair an invalid choice.
"""

import json
from typing import cast

import jax
import numpy as np
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    MOVEMENT_ACTION_NAME_BY_ID,
    TARGET_ACTION_NAME_BY_ID,
)
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.policies.actor import ActorAction

MOVE_NAMES = tuple(name.lower() for name in MOVEMENT_ACTION_NAME_BY_ID)
TARGET_NAMES = tuple(
    name.lower().replace(" ", "_") for name in TARGET_ACTION_NAME_BY_ID
)
COMBAT_NAMES = tuple(
    (
        "no_combat" if target == 0 else f"{name}_basic",
        "ultimate" if target == 0 else f"{name}_ultimate",
    )
    for target, name in enumerate(TARGET_NAMES)
)
_COMBAT_IDS = {
    name: (target, ultimate)
    for target, pair in enumerate(COMBAT_NAMES)
    for ultimate, name in enumerate(pair)
}


class ReplyFormatError(ValueError):
    """The reply does not follow the chosen text format; no action was repaired."""


class IllegalActionError(ValueError):
    """A decoded action has invalid native categories or is masked this turn."""


class InvalidActionError(IllegalActionError):
    """A custom parser returned the wrong record, shape, dtype or native category."""


class MaskedActionError(IllegalActionError):
    """A well-formed native action is forbidden by the current actor masks."""


def _host_masks(masks: ActionMask) -> ActionMask:
    """Copy one actor's masks to host and check Boolean shapes and agreement.

    Parameters
    ----------
    masks : ActionMask
        Same-epoch scalar actor masks, with shapes (9,), (11,), (2,), (11, 2).

    Returns
    -------
    ActionMask
        Host NumPy leaves. Inputs are unchanged; device inputs transfer once.

    Raises
    ------
    ValueError
        Mask shapes, Boolean dtypes or marginal/joint agreement are invalid.
    """
    if not isinstance(masks, ActionMask):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Masks must be one actor's ActionMask")
    host = jax.device_get(masks)
    for name, value, shape in zip(
        masks._fields, host, ((9,), (11,), (2,), (11, 2)), strict=True
    ):
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.bool_:
            raise ValueError(f"{name} must be Boolean with shape {shape}")
    joint = np.asarray(host.select_target_use_ultimate_joint_mask)
    if not np.array_equal(
        host.select_target_mask, joint.any(axis=1)
    ) or not np.array_equal(host.use_ultimate_mask, joint.any(axis=0)):
        raise ValueError("Combat marginal masks must agree with the joint mask")
    if not np.any(host.move_mask) or not joint.any():
        raise ValueError("Actor masks must admit a movement and a combat pair")
    return host


def legal_action_names(masks: ActionMask) -> dict[str, tuple[str, ...]]:
    """List current legal move and complete combat names in native category order.

    Parameters
    ----------
    masks : ActionMask
        One actor's Boolean masks in the frame used for the prompt.

    Returns
    -------
    dict
        Keys ``move`` and ``combat`` each hold a tuple of legal strings. Any
        listed move can accompany any listed combat pair. Names never change
        meaning when the legal subset changes. ``no_combat`` is (0, 0);
        ``ultimate`` is the untargeted (0, 1) pair when permitted.

    Raises
    ------
    ValueError
        Masks violate the scalar actor shape, dtype or joint-mask contract.
    """
    host = _host_masks(masks)
    return {
        "move": tuple(
            name
            for name, allowed in zip(MOVE_NAMES, host.move_mask, strict=True)
            if allowed
        ),
        "combat": tuple(
            COMBAT_NAMES[target][ultimate]
            for target in range(11)
            for ultimate in range(2)
            if host.select_target_use_ultimate_joint_mask[target, ultimate]
        ),
    }


def validate_actor_action(action: ActorAction, masks: ActionMask) -> ActorAction:
    """Check one native action against the original current masks without repair.

    Parameters
    ----------
    action : ActorAction
        Scalar integer move, target and Ultimate categories. Booleans, floats,
        arrays with leading axes and out-of-range integers are rejected before
        conversion, so overflow cannot turn a bad value into a legal choice.
    masks : ActionMask
        Retained original masks for this actor, epoch and coordinate frame.
        The complete target/Ultimate pair, not just its margins, is checked.

    Returns
    -------
    ActorAction
        Three scalar host NumPy int32 leaves. Inputs remain unchanged. The caller
        still owns correct actor/epoch routing and any world-frame conversion.

    Raises
    ------
    IllegalActionError
        InvalidActionError means a bad record, shape, dtype or category.
        MaskedActionError means a valid native category is currently forbidden.
        Both retain IllegalActionError as their public base.
    ValueError
        The masks themselves violate their contract.
    """
    if not isinstance(action, ActorAction):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidActionError("The parser must return an ActorAction")
    host = _host_masks(masks)
    values: list[int] = []
    for name, value, size in zip(
        action._fields, jax.device_get(action), (9, 11, 2), strict=True
    ):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "iu":
            raise InvalidActionError(f"{name} must be a scalar integer category")
        category = int(array)
        if not 0 <= category < size:
            raise InvalidActionError(f"{name} must be in 0..{size - 1}")
        values.append(category)
    move, target, ultimate = values
    if not host.move_mask[move]:
        raise MaskedActionError(f"Movement {MOVE_NAMES[move]} is not legal this turn")
    if not host.select_target_use_ultimate_joint_mask[target, ultimate]:
        raise MaskedActionError(
            f"Combat {COMBAT_NAMES[target][ultimate]} is not legal this turn"
        )
    return ActorAction(
        *(cast(Array, np.asarray(value, dtype=np.int32)) for value in values)
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a decoded JSON object, rejecting any repeated key in its pairs."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ReplyFormatError(f"Reply repeats the key {name!r}")
        result[name] = value
    return result


def parse_action_reply(reply: str, masks: ActionMask) -> ActorAction:
    """Decode exactly one named JSON move/combat reply and check its legality.

    Parameters
    ----------
    reply : str
        One JSON object with exactly two string values: ``move`` and ``combat``.
        Surrounding whitespace is allowed; fences, prose, duplicate keys, unknown
        names and extra keys are errors. No spelling repair or retry occurs.
    masks : ActionMask
        Original current masks for the same actor and prompt frame.

    Returns
    -------
    ActorAction
        Validated scalar host int32 heads in the prompt frame. This helper does
        not mirror movement; its caller performs any reflection exactly once.

    Raises
    ------
    ReplyFormatError
        The reply is not the exact named JSON format.
    IllegalActionError
        The named choice is currently masked.
    ValueError
        The masks themselves violate their contract.
    """
    if not isinstance(reply, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ReplyFormatError("Reply must be JSON text")
    try:
        decoded: object = json.loads(reply, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ReplyFormatError("Reply must contain exactly one JSON object") from exc
    if not isinstance(decoded, dict):
        raise ReplyFormatError("Reply must contain exactly move and combat")
    fields = cast(dict[str, object], decoded)
    if set(fields) != {"move", "combat"}:
        raise ReplyFormatError("Reply must contain exactly move and combat")
    move, combat = fields["move"], fields["combat"]
    if not isinstance(move, str) or move not in MOVE_NAMES:
        raise ReplyFormatError("Reply has an unknown movement name")
    if not isinstance(combat, str) or combat not in _COMBAT_IDS:
        raise ReplyFormatError("Reply has an unknown combat name")
    target, ultimate = _COMBAT_IDS[combat]
    action = ActorAction(
        *(
            cast(Array, np.asarray(value, np.int32))
            for value in (MOVE_NAMES.index(move), target, ultimate)
        )
    )
    return validate_actor_action(action, masks)
