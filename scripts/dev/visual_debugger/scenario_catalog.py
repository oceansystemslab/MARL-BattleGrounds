"""List debugger scenario metadata without importing the simulator or JAX.

Launchers and shell completion use this catalog to discover names and summaries.
The live scenario registry in ``scenarios`` uses these same metadata rows. Helpers
return ordered immutable tuples and do not create sessions, read artifacts, or
write files.
"""

from dataclasses import dataclass
from typing import Literal

type ScenarioMode = Literal["interactive", "scripted"]
type ScenarioAudience = Literal["researcher", "stress"]


@dataclass(frozen=True, slots=True)
class ScenarioCatalogEntry:
    """Describe one registered debugger scenario without constructing it.

    Attributes
    ----------
    name : str
        Stable launcher key.
    title : str
        Human-readable scenario title.
    description : str
        Short description of the demonstration.
    mode : {"interactive", "scripted"}
        Whether actors are controlled live or follow saved scripted commands.
    default_controlled_slot : int
        Initial zero-based global actor slot.
    audience : {"researcher", "stress"}
        Whether discovery includes the scenario by default or only after stress opt-in.
    """

    name: str
    title: str
    description: str
    mode: ScenarioMode
    default_controlled_slot: int
    audience: ScenarioAudience

    def summary(self) -> str:
        """Format one stable line for scenario listings.

        Returns
        -------
        str
            Padded launcher name and mode followed by the short description.
        """
        return f"{self.name:<22} {self.mode:<11} {self.description}"


RESEARCHER_SCENARIO_CATALOG: tuple[ScenarioCatalogEntry, ...] = (
    ScenarioCatalogEntry(
        name="arena_5v5",
        title="5v5 geometry and combat laboratory",
        description=(
            "Interactive LOS, visibility, range, relation, and mask inspection."
        ),
        mode="interactive",
        default_controlled_slot=0,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="basic_support",
        title="Basic damage and support",
        description="Scripted simultaneous Basic damage, healing, and passives.",
        mode="scripted",
        default_controlled_slot=0,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="ultimate_showcase",
        title="Five-class Ultimate showcase",
        description="Scripted activation and lifecycle of all class Ultimates.",
        mode="scripted",
        default_controlled_slot=0,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="aura_crossfire",
        title="Aura crossfire",
        description="Scripted reciprocal Basics under both aura families.",
        mode="scripted",
        default_controlled_slot=2,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="stacked_team_auras",
        title="Stacked team auras",
        description=(
            "Two same-team Mage and Warrior emitters stack on reciprocal Basics."
        ),
        mode="scripted",
        default_controlled_slot=4,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="status_stack",
        title="Status composition and lifecycle",
        description="Scripted stacked control, mitigation, break, and movement.",
        mode="scripted",
        default_controlled_slot=5,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="team_focus_crossfire",
        title="Focus fire and coordinated healing",
        description=(
            "Repeated and simultaneous damage, healing, Crippling Poison, and "
            "Holy Word: Salvation."
        ),
        mode="scripted",
        default_controlled_slot=2,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="mirrored_ultimates",
        title="Mirrored five-class Ultimates",
        description="Reciprocal and mirrored activation of all Ultimate families.",
        mode="scripted",
        default_controlled_slot=0,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="death_respawn_cycle",
        title="Death, respawn, and spawn shield",
        description="A complete lethal, corpse, wave, respawn, and shield lifecycle.",
        mode="scripted",
        default_controlled_slot=5,
        audience="researcher",
    ),
    ScenarioCatalogEntry(
        name="recovery_refresh_cycle",
        title="Recovery, refresh, and reapplication",
        description="Regeneration, readiness, rejection, refresh, break, and expiry.",
        mode="scripted",
        default_controlled_slot=0,
        audience="researcher",
    ),
)

STRESS_SCENARIO_CATALOG: tuple[ScenarioCatalogEntry, ...] = (
    ScenarioCatalogEntry(
        name="moving_basic_crossfire",
        title="Moving Basic crossfire",
        description="Reciprocal Basics and healing across moving successor anchors.",
        mode="scripted",
        default_controlled_slot=0,
        audience="stress",
    ),
    ScenarioCatalogEntry(
        name="moving_focus_crossfire",
        title="Moving focus crossfire",
        description="Moving focus fire and healing converge on one recipient.",
        mode="scripted",
        default_controlled_slot=2,
        audience="stress",
    ),
    ScenarioCatalogEntry(
        name="charge_convergence",
        title="Converging Charge routes",
        description="Three simultaneous reciprocal and shared-target Charges.",
        mode="scripted",
        default_controlled_slot=0,
        audience="stress",
    ),
    ScenarioCatalogEntry(
        name="trap_lifecycle",
        title="Freezing Trap lifecycle stress",
        description=(
            "Exact application, damage break, reapplication, and age-to-zero "
            "status lifecycle."
        ),
        mode="scripted",
        default_controlled_slot=0,
        audience="stress",
    ),
    ScenarioCatalogEntry(
        name="max_status_stack",
        title="Maximum status density",
        description="All nine compatible status channels on one recipient.",
        mode="scripted",
        default_controlled_slot=0,
        audience="stress",
    ),
    ScenarioCatalogEntry(
        name="lifecycle_density",
        title="Lifecycle density stress",
        description=(
            "Concurrent lethal clearing, recovery, readiness, respawn, and "
            "Spawn Shield expiry."
        ),
        mode="scripted",
        default_controlled_slot=5,
        audience="stress",
    ),
)

SCENARIO_CATALOG: tuple[ScenarioCatalogEntry, ...] = (
    *RESEARCHER_SCENARIO_CATALOG,
    *STRESS_SCENARIO_CATALOG,
)
SCENARIO_CATALOG_BY_NAME = {entry.name: entry for entry in SCENARIO_CATALOG}


def iter_scenario_catalog(
    *,
    include_stress: bool = False,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Return registered scenario metadata in the declared display order.

    Parameters
    ----------
    include_stress : bool, optional
        False returns researcher scenarios only. True appends stress scenarios.

    Returns
    -------
    tuple of ScenarioCatalogEntry
        Immutable metadata rows without constructing simulator state.
    """
    return SCENARIO_CATALOG if include_stress else RESEARCHER_SCENARIO_CATALOG


def iter_scenario_summaries(*, include_stress: bool = False) -> tuple[str, ...]:
    """Return the selected scenario list as readable launcher lines.

    Parameters
    ----------
    include_stress : bool, optional
        False omits stress scenarios; True includes them after researcher scenarios.

    Returns
    -------
    tuple of str
        One formatted line per catalog entry in the same declared order.
    """
    return tuple(
        entry.summary()
        for entry in iter_scenario_catalog(include_stress=include_stress)
    )


__all__ = [
    "RESEARCHER_SCENARIO_CATALOG",
    "SCENARIO_CATALOG",
    "SCENARIO_CATALOG_BY_NAME",
    "STRESS_SCENARIO_CATALOG",
    "ScenarioCatalogEntry",
    "iter_scenario_catalog",
    "iter_scenario_summaries",
]
