"""Standard-library-only metadata for registered debugger scenarios.

Live scenario constructors intentionally remain in :mod:`scenarios`.  Keeping
their launch metadata here lets discovery and shell completion run on machines
without initializing an array backend, while the live registry consumes this
same catalog as its single metadata authority.
"""

from dataclasses import dataclass
from typing import Literal

type ScenarioMode = Literal["interactive", "scripted"]
type ScenarioAudience = Literal["researcher", "stress"]


@dataclass(frozen=True, slots=True)
class ScenarioCatalogEntry:
    """Backend-free launch metadata for one registered scenario."""

    name: str
    title: str
    description: str
    mode: ScenarioMode
    default_controlled_slot: int
    audience: ScenarioAudience

    def summary(self) -> str:
        """Return the stable one-line launcher representation."""
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
    """Return stable launch metadata without importing live simulator modules."""
    return SCENARIO_CATALOG if include_stress else RESEARCHER_SCENARIO_CATALOG


def iter_scenario_summaries(*, include_stress: bool = False) -> tuple[str, ...]:
    """Return stable one-line scenario summaries without backend imports."""
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
