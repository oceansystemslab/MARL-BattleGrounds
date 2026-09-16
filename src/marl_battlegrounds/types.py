"""Import the existing simulator and actor types from one public location.

These are the original NamedTuple classes, not copies or adapters. Importing
``Action`` here gives the same class as importing it from Core. Their arrays
remain dynamic JAX data. Constructing a NamedTuple packages values; it does not
validate shapes, dtypes, physical validity or action legality.

Available types:
    Action: Joint ``move``, ``select_target`` and ``use_ultimate`` int32 arrays.
        Each field has shape ``(10,)`` for a scalar environment or ``(B, 10)``
        for a native batch. Slots 0..4 belong to Team A and 5..9 to Team B.
        Target categories keep their actor-relative meaning in both teams.
    ActorAction: The same three action fields for one actor, each shape ``()``
        and dtype int32. Category ranges are move 0..8, target 0..10 and
        Ultimate 0..1. A category within range may still be masked out.
    ActionMask: Bool arrays describing current allowed choices. For one actor,
        move, target and Ultimate masks have shapes ``(9,)``, ``(11,)`` and
        ``(2,)``; their joint target/Ultimate mask has shape ``(11, 2)``.
        Scalar environment masks add the leading ten-actor axis; native masks
        add B before that axis. The joint mask owns exact combat-pair legality.
    EnvConfig: Existing resolved simulator settings and agent profile. Ordinary
        task factories return scalar settings with fixed-size array fields.
        Public native preparation adds B to every leaf. Ordered spawn banks
        have shape ``(2, 5, 2)`` per game; profiles have ten slot-aligned entries.
        Use task factories or validated exact configs before compiled execution.
    DoneFlags: Bool ``terminated`` and ``truncated`` values, shape ``()`` for a
        scalar game or ``(B,)`` for a native batch. ``done`` returns their OR.
        These mark episode completion; an individual agent death is separate.
    Observation: Existing named feature, visibility, action-history and spawn
        lifecycle families. Scalar environment leaves lead with ten actors;
        native leaves lead with B then ten actors. For a scalar space check,
        select one actor from one game; a native batch needs both indices.
        Feature values are float32, masks bool, and index/count values int32.
        ``env.observation_space(agent)`` describes each remaining field shape
        without flattening it or adding identity features.

Import these names explicitly, for example
``from marl_battlegrounds.types import Action, ActorAction``. They are not
package-root exports such as ``marl_bgs.Action``.

Use ``env.action_space(agent).contains(value)`` for host-side structural checks
of one ActorAction. It may copy device data and is not for a JAX rollout loop.
Use the actual ActionMask for legal choices and ``env.sample_actions`` for the
optional legal sampler. Runner EnvironmentState and privileged step diagnostics
are separate from an actor's Observation and are not exported as Core types here.
"""

from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction

__all__ = [
    "Action",
    "ActionMask",
    "ActorAction",
    "DoneFlags",
    "EnvConfig",
    "Observation",
]
