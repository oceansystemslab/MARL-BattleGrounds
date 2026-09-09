"""Generate the researcher data dictionary from the fixed scalar catalog."""

import argparse
import csv
import io
from pathlib import Path

from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS, STATUS_LABELS
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS

_IDENTITY_MEANINGS = {
    "run_id": (
        "Unique run identity; combine with phase, pass_id and episode_id to join runs."
    ),
    "phase": (
        "Research phase declared by the caller, such as training, validation, "
        "evaluation or tournament."
    ),
    "pass_id": "Caller-declared pass identity within this run and phase.",
    "episode_id": (
        "Positive one-based episode identity assigned before execution, "
        "local to this pass."
    ),
    "seed_id": (
        "Scheduled random-stream identity; independent of completion order "
        "and batch size."
    ),
    "map_id": (
        "Registered map identity; detailed geometry is stored once in run_details.json."
    ),
    "config_id": (
        "SHA-256 identity of the resolved episode configuration "
        "stored in run_details.json."
    ),
    "team_a_policy": (
        "Recorded Team A policy name; variables and checkpoint provenance "
        "are stored in pass details."
    ),
    "team_b_policy": (
        "Recorded Team B policy name; variables and checkpoint provenance "
        "are stored in pass details."
    ),
    "checkpoint_id": (
        "Optional pass checkpoint identity; blank for evolving training policies "
        "or unspecified checkpoints."
    ),
}


def dictionary_csv() -> str:
    """List every run-table column in its exported order, including identity."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "column",
            "meaning",
            "units",
            "scope",
            "subjects",
            "missing_when",
            "included_in",
            "gui_label",
            "direction",
            "required_class_id",
            "status",
        )
    )
    for name in IDENTITY_COLUMNS:
        agent = name.startswith("agent_")
        active = name.endswith("_active")
        subject = name.split("_")[1] if agent else ""
        meaning = (
            "One when this fixed slot is active in the episode roster, otherwise zero."
            if agent and active
            else (
                "Configured class ID for this fixed slot: Neutral=0, Mage=1, "
                "Warrior=2, Hunter=3, Rogue=4, Priest=5. "
                "Read together with its active flag."
            )
            if agent
            else _IDENTITY_MEANINGS[name]
        )
        writer.writerow(
            (
                name,
                meaning,
                "indicator" if agent and active else "identifier",
                "agent" if agent else "episode identity",
                subject,
                "Not supplied."
                if name
                in {
                    "seed_id",
                    "map_id",
                    "checkpoint_id",
                    "team_a_policy",
                    "team_b_policy",
                }
                else "Never in a completed row.",
                "priority;full",
                "Active"
                if agent and active
                else "Class"
                if agent
                else "Episode Details",
                "No Preferred Direction",
                "",
                "",
            )
        )
    for column in METRIC_COLUMNS:
        writer.writerow(
            (
                column.name,
                column.description,
                column.unit,
                column.scope,
                " / ".join(str(subject) for subject in column.subjects),
                column.missing_when,
                "priority;full" if column.priority else "full",
                column.label,
                {
                    "higher": "Higher Is Better",
                    "lower": "Lower Is Better",
                    "descriptive": "No Preferred Direction",
                }[column.direction],
                column.required_class_id or "",
                ""
                if column.status_channel is None
                else STATUS_LABELS[column.status_channel],
            )
        )
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/evaluation/metric_columns.csv")
    )
    args = parser.parse_args()
    expected = dictionary_csv()
    if args.check:
        if args.output.read_text() != expected:
            parser.error("metric dictionary is stale; rerun without --check")
    else:
        args.output.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
