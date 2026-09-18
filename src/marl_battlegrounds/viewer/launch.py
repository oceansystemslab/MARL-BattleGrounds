"""Launch the shared Viewer from a replay path or a validated loaded bundle.

Browser launch uses installed package resources and a loopback server. Repository
launchers may supply their editable frontend directory. Loading and validation
finish before the server starts. This module does not build simulator scenarios.
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer.options import PlaybackOptions, validate_playback_options

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle


def launch_replay(
    path: str | Path | None,
    *,
    options: PlaybackOptions,
    loaded_bundle: LoadedReplayBundle | None = None,
    asset_root: Path | None = None,
) -> int:
    """Open a validated replay in the browser or optional static renderer.

    Parameters
    ----------
    path : str, pathlib.Path or None
        Saved replay to load once. Use None when passing ``loaded_bundle``.
    options : PlaybackOptions
        Shared resolved playback settings. Static mode needs an explicit frame.
    loaded_bundle : LoadedReplayBundle or None, optional
        Already validated replay for repository samples. It is not loaded again.
        Supply exactly one of path and this bundle.
    asset_root : pathlib.Path or None, optional
        Developer-only editable frontend directory. None uses package resources,
        keeping any temporary extraction alive for the complete server lifetime.

    Returns
    -------
    int
        The browser server or static renderer exit status.

    Raises
    ------
    ValueError
        If source selection, options, replay contents or authorization are invalid.
    OSError
        If replay/resources cannot be read or the local server cannot bind.
    ImportError
        If optional static plotting dependencies are unavailable.

    Notes
    -----
    Browser mode may open a browser and blocks until server shutdown. Static mode
    opens a Matplotlib window. Neither mode advances the environment or changes
    replay files. Existing requested numerical analysis remains lazy.
    """
    validate_playback_options(options)
    if (path is None) == (loaded_bundle is None):
        raise ValueError("Supply exactly one replay path or loaded bundle.")
    if options.static:
        from marl_battlegrounds.viewer.static import (
            run_static_replay_artifact_renderer,
            run_static_replay_renderer,
        )

        assert options.frame_index is not None
        if loaded_bundle is not None:
            return run_static_replay_artifact_renderer(
                replay=loaded_bundle.replay,
                frame_index=options.frame_index,
                show_ranges=options.ranges,
            )
        assert path is not None
        return run_static_replay_renderer(
            replay_path=Path(path),
            frame_index=options.frame_index,
            show_ranges=options.ranges,
        )

    bundle = loaded_bundle
    if bundle is None:
        from marl_battlegrounds.evaluation.replay_io import ReplayLoadError, load_replay

        assert path is not None
        try:
            bundle = load_replay(path)
        except ReplayLoadError as exc:
            raise ValueError(f"Replay could not be loaded: {exc}") from exc

    from marl_battlegrounds.viewer.replay_protocol import (
        ReplayApiErrorV1,
        ReplayCommandRequestV1,
    )
    from marl_battlegrounds.viewer.replay_service import ReplayViewerService
    from marl_battlegrounds.viewer.server import (
        REPLAY_HTTP_ROUTES,
        HttpCoordinatorBinding,
        serve_browser_debugger,
    )

    service = ReplayViewerService(
        bundle,
        initial_frame_index=0 if options.frame_index is None else options.frame_index,
        view_mode=options.view,
        pov_global_slot=options.pov_slot,
        preset="analysis",
        show_ranges=options.ranges,
        verbose=False,
    )
    coordinator = HttpCoordinatorBinding(
        mode="replay",
        initial_show_ranges=options.ranges,
        routes=REPLAY_HTTP_ROUTES,
        request_model=ReplayCommandRequestV1,
        error_factory=ReplayApiErrorV1,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=service.current_timeline,
        current_presentation=service.current_presentation,
        current_metric_report=service.current_metric_report,
        metric_analysis=service.metric_analysis,
        metric_catalog=service.metric_catalog,
        episode_details=service.episode_details,
    )
    resource_context = (
        nullcontext(asset_root)
        if asset_root is not None
        else as_file(files("marl_battlegrounds.viewer").joinpath("web"))
    )
    with resource_context as root:
        return serve_browser_debugger(
            service,
            asset_root=root,
            port=options.port,
            open_browser=not options.no_open,
            coordinator=coordinator,
        )
