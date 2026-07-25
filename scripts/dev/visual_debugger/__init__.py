"""Public entry surface for the repository-local visual debugger."""

from scripts.dev.visual_debugger.scenarios import (
    get_scenario,
    iter_scenario_summaries,
    list_scenarios,
)


def run_visual_debugger(
    *,
    scenario_name: str,
    seed: int,
    controlled_global_slot: int | None,
    static: bool,
    verbose: bool,
    show_ranges: bool,
    include_stress: bool = False,
) -> int:
    """Load the temporary Matplotlib client only when it is selected."""
    from scripts.dev.visual_debugger.app import (
        run_visual_debugger as run_matplotlib_debugger,
    )

    return run_matplotlib_debugger(
        scenario_name=scenario_name,
        seed=seed,
        controlled_global_slot=controlled_global_slot,
        static=static,
        verbose=verbose,
        show_ranges=show_ranges,
        include_stress=include_stress,
    )


__all__ = [
    "get_scenario",
    "iter_scenario_summaries",
    "list_scenarios",
    "run_visual_debugger",
]
