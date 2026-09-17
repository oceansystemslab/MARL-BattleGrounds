"""Resolve shared browser and static replay options without loading a backend.

The package command and repository launcher use the same defaults and option
checks. Source selection remains with each launcher; this module reads no files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, cast

PLAYBACK_OPTION_LABELS = (
    ("frame_index", "--frame-index"),
    ("pov_slot", "--pov-slot"),
    ("static", "--static"),
    ("no_open", "--no-open"),
    ("port", "--port"),
    ("view", "--view"),
    ("ranges", "--ranges/--no-ranges"),
)


@dataclass(frozen=True, slots=True)
class PlaybackOptions:
    """Immutable replay display values, including explicitly supplied fields.

    ``frame_index=None`` uses browser frame zero; static display requires an
    explicit index. ``pov_slot`` is a global actor slot, or None for the existing
    service default. ``view`` is the wire authority, researcher or pov. Browser
    defaults open a local window, use an available port and hide range overlays.
    ``supplied`` stores field names, so an explicitly supplied default still
    receives the same compatibility checks as any other supplied value.
    """

    frame_index: int | None = None
    pov_slot: int | None = None
    static: bool = False
    no_open: bool = False
    port: int = 0
    view: Literal["researcher", "pov"] = "researcher"
    ranges: bool = False
    supplied: frozenset[str] = frozenset()


def resolve_playback_options(namespace: argparse.Namespace) -> PlaybackOptions:
    """Fill display defaults while preserving the supplied playback fields.

    Parameters
    ----------
    namespace : argparse.Namespace
        Parsed fields. Omitted flags must use ``argparse.SUPPRESS``. The public
        view name oracle maps to the existing researcher wire authority.

    Returns
    -------
    PlaybackOptions
        New immutable display options. No replay, backend or server is loaded.
    """
    view = getattr(namespace, "view", "researcher")
    return PlaybackOptions(
        frame_index=cast(int | None, getattr(namespace, "frame_index", None)),
        pov_slot=cast(int | None, getattr(namespace, "pov_slot", None)),
        static=cast(bool, getattr(namespace, "static", False)),
        no_open=cast(bool, getattr(namespace, "no_open", False)),
        port=cast(int, getattr(namespace, "port", 0)),
        view=cast(
            Literal["researcher", "pov"], "researcher" if view == "oracle" else view
        ),
        ranges=cast(bool, getattr(namespace, "ranges", False)),
        supplied=frozenset(vars(namespace))
        & frozenset(name for name, _label in PLAYBACK_OPTION_LABELS),
    )


def validate_playback_options(options: PlaybackOptions) -> None:
    """Reject browser-only options in a static replay request.

    Parameters
    ----------
    options : PlaybackOptions
        Resolved values and exact supplied fields. Numerical ranges and replay
        authorization remain with the replay service and renderer.

    Raises
    ------
    ValueError
        If static mode lacks an explicit frame or includes browser-only flags.
    """
    if not options.static:
        return
    if "frame_index" not in options.supplied:
        raise ValueError("--frame-index is required with --static.")
    forbidden = options.supplied - {"static", "frame_index", "ranges"}
    for name, label in PLAYBACK_OPTION_LABELS:
        if name in forbidden:
            raise ValueError(f"{label} is unavailable with --static.")
