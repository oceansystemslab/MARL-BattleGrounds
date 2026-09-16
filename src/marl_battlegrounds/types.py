"""Import simulator, actor and System types from one public location.

These names refer to their original owning classes. Importing ``Action`` here
gives the same class as importing it from Core. Core and actor types load with
this module; System and recording types load only when requested. Their numerical
arrays stay dynamic JAX data. Core NamedTuple constructors store values without
validating shapes, dtypes, physical validity or action legality. Each System type
documents its own validation and host/JAX contract.

Available types:
    Action: Joint ``move``, ``select_target`` and ``use_ultimate`` int32 arrays.
        Each field has shape ``(10,)`` for a scalar environment or ``(B, 10)``
        for a native batch. Slots 0..4 belong to Team A and 5..9 to Team B.
        Target categories keep their actor-relative meaning in both teams.
    ActorAction: The same three int32 action fields. A scalar Policy uses shape
        ``()`` per field; a team adds five slots, and a System uses ``(B, 5)``.
        Category ranges are move 0..8, target 0..10 and Ultimate 0..1. A category
        within range may still be masked out.
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
    System: Immutable method description with explicit parameters, initialization
        and reset hooks. It adds no actor information rights or hidden RNG.
    SystemInput: One team's five separate permitted actor views, masks and
        lifecycle flags, always with a leading environment axis B.
    SystemOutput: Frozen method result holding submitted actions, next memory,
        optional learning values and optional component choices.
    SystemState: The two methods' memories, reset bindings, saved initialization
        roots and last numerical trace. Obtain it through init_systems.
    SystemStepData: One team's submitted actions, rewards, configured activity,
        pre-step episode identities and real-transition flags, with leading B.
    EpisodeTrackingState: Immutable episode/source bindings and checked stage counts.
        Carry it beside the matching environment state; stage checks are host-only.
    FinalEpisodeData: Compact pre-reset observations, masks and old episode identity.
        Its valid mask selects completed games; privileged state is opt-in.
    EpisodeStartRecords: Compact numerical first-transition source claims. The
        writer verifies them before treating source relationships as known.
    PolicyTrace: Decision identities and reported component choices. This is
        numerical runner data, not actor input or an automatic recording stream.

Import these names explicitly, for example
``from marl_battlegrounds.types import Action, ActorAction``. Core and actor types
are not package-root exports such as ``marl_bgs.Action``. The System types also
have package-root exports, except PolicyTrace.

Use ``env.action_space(agent).contains(value)`` for host-side structural checks
of one ActorAction. It may copy device data and is not for a JAX rollout loop.
Use the actual ActionMask for legal choices and ``env.sample_actions`` for the
optional legal sampler. Runner EnvironmentState and privileged step diagnostics
are separate from an actor's Observation and are not exported as Core types here.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction

if TYPE_CHECKING:
    from marl_battlegrounds.autoreset import FinalEpisodeData
    from marl_battlegrounds.episode_tracking import EpisodeTrackingState
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTrace,
        System,
        SystemInput,
        SystemOutput,
        SystemState,
        SystemStepData,
    )
    from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords

__all__ = [
    "Action",
    "ActionMask",
    "ActorAction",
    "DoneFlags",
    "EnvConfig",
    "EpisodeStartRecords",
    "EpisodeTrackingState",
    "FinalEpisodeData",
    "Observation",
    "PolicyTrace",
    "System",
    "SystemInput",
    "SystemOutput",
    "SystemState",
    "SystemStepData",
]


def __getattr__(name: str) -> object:
    """Load and cache an original System or recording type when requested.

    Parameters
    ----------
    name : str
        Exact lazy type name listed in this module's __all__.

    Returns
    -------
    object
        The class owned by policy_execution or recording_types. Later access reuses
        the same cached object; no wrapper or duplicate class is constructed.

    Raises
    ------
    AttributeError
        name is not one of the supported lazy types.

    Notes
    -----
    Loading a System type imports its method-execution module and dependencies.
    Import failures propagate to the caller. Existing Core and actor classes
    are already present and do not use this fallback.
    """
    if name in {"FinalEpisodeData", "EpisodeTrackingState"}:
        module = "autoreset" if name == "FinalEpisodeData" else "episode_tracking"
        value = getattr(import_module(f"marl_battlegrounds.{module}"), name)
        globals()[name] = value
        return value
    if name == "EpisodeStartRecords":
        value = getattr(
            import_module("marl_battlegrounds.evaluation.recording_types"), name
        )
        globals()[name] = value
        return value
    if name in {
        "PolicyTrace",
        "System",
        "SystemInput",
        "SystemOutput",
        "SystemState",
        "SystemStepData",
    }:
        value = getattr(
            import_module("marl_battlegrounds.evaluation.policy_execution"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
