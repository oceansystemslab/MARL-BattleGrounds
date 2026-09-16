"""Generate metric-column documentation from the shared metric catalog.

Run ``python -m scripts.dev.export_metric_dictionary`` at the repository root to
update the CSV dictionary and generated specification tables. Add ``--check`` to
compare existing text without writing. The catalog remains the measurement
naming/meaning authority; this module formats its data rather than computing
metrics or replaying episodes.
"""

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    METRIC_TOPICS,
    METRIC_TOPICS_BY_NAME,
    METRIC_VIEW_LABELS,
    STATUS_LABELS,
    metric_guidance,
    metric_locations,
    metric_primary_location,
    metric_topic_text,
)
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS

_SUMMARY_START = "<!-- metric-family-summary:start -->"
_SUMMARY_END = "<!-- metric-family-summary:end -->"
_NAVIGATION_START = "<!-- metric-navigation:start -->"
_NAVIGATION_END = "<!-- metric-navigation:end -->"
_SPECIFICATION = Path("docs/evaluation/metric_specification.md")

# Count each column once in the paper, even if several viewer groups show it.
_MANUSCRIPT_FAMILIES = (
    (
        "Episode results",
        ("priority",),
        "Episode length, outcomes, returns, scores, kills, and deaths.",
    ),
    (
        "Ability use and action acceptance",
        ("abilities", "action_acceptance"),
        "How often abilities were used, who they targeted, "
        "and which actions the game rejected.",
    ),
    (
        "Deaths and respawning",
        ("deaths", "respawn"),
        "Who died, how long agents were dead, and when they returned.",
    ),
    (
        "Kills and coordination",
        ("kill_contributions", "coordination"),
        "Who helped kill each enemy, who killed alone, and whether attackers "
        "chose the same target.",
    ),
    (
        "Damage",
        ("damage_done", "recipient_damage", "damage_received"),
        "Who dealt damage, who took it, which ability dealt it, "
        "and each agent's share.",
    ),
    (
        "Healing and excess",
        ("healing_done", "recipient_healing", "healing_received", "excess_healing"),
        "Who healed whom, which ability was used, how much healing was useful "
        "or excess, and automatic health recovery.",
    ),
    (
        "Effects on controlled recipients",
        ("controlled_damage", "controlled_healing", "controlled_kills"),
        "Damage, healing, and kill credit when the affected agent already "
        "had a named harmful status.",
    ),
    (
        "Status applications, duration, and Freedom",
        ("status_applications", "status_active_steps", "freedom"),
        "Who applied each status, who had it and for how long, "
        "and when Freedom protected movement.",
    ),
    (
        "Trap breaks",
        ("trap_breaks",),
        "Who was trapped, who broke each Trap with damage, "
        "and how much Trap time remained.",
    ),
    (
        "Burst",
        ("burst",),
        "Damage dealt during Burst, who took it, and who helped "
        "with kills while Burst was active.",
    ),
    (
        "Auras",
        ("aura_coverage", "aura_benefits"),
        "Who gave and received aura coverage, extra damage from Mage auras, "
        "and damage blocked by Warrior auras.",
    ),
    (
        "Poison healing prevention",
        ("poison",),
        "Priest healing prevented on affected agents and teams.",
    ),
    (
        "Lethal-damage rescues",
        ("priest_rescue",),
        "Who could be saved from lethal damage, who survived, "
        "and which healers and abilities helped.",
    ),
    (
        "Formation",
        ("formation",),
        "Distances between living teammates and how many times each pair was measured.",
    ),
)


def family_summary_markdown() -> str:
    """Render the paper's measurement-family table with each column counted once.

    Returns
    -------
    str
        Markdown table wrapped in its generated-block markers.

    Raises
    ------
    ValueError
        If manuscript families duplicate or omit a metric catalog family.
    """
    counts = Counter(column.family for column in METRIC_COLUMNS)
    assigned = [
        family for _, families, _ in _MANUSCRIPT_FAMILIES for family in families
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(counts):
        raise ValueError(
            "manuscript families must cover the metric catalog exactly once"
        )
    lines = [
        _SUMMARY_START,
        "| Measurement family | Numerical columns | What the family describes |",
        "| --- | ---: | --- |",
    ]
    for label, families, description in _MANUSCRIPT_FAMILIES:
        count = sum(counts[family] for family in families)
        lines.append(f"| {label} | {count:,} | {description} |")
    lines.extend(
        (
            f"| **Total** | **{len(METRIC_COLUMNS):,}** | **Unique numerical columns; "
            "shared viewer appearances are counted once.** |",
            _SUMMARY_END,
        )
    )
    return "\n".join(lines)


def specification_with_summary(text: str) -> str:
    """Replace generated family and navigation tables inside existing Markdown.

    Parameters
    ----------
    text : str
        Complete specification text with one ordered marker pair for each table.

    Returns
    -------
    str
        Updated text with all content outside the generated blocks preserved.

    Raises
    ------
    ValueError
        If markers are missing, repeated or reversed, or family coverage is invalid.

    Notes
    -----
    This transforms a string only. The caller owns reading and writing the file.
    """
    if text.count(_SUMMARY_START) != 1 or text.count(_SUMMARY_END) != 1:
        raise ValueError("metric specification must contain one family-summary block")
    start = text.index(_SUMMARY_START)
    end = text.index(_SUMMARY_END) + len(_SUMMARY_END)
    if end < start:
        raise ValueError("metric family-summary markers are reversed")
    text = text[:start] + family_summary_markdown() + text[end:]
    if text.count(_NAVIGATION_START) != 1 or text.count(_NAVIGATION_END) != 1:
        raise ValueError("metric specification must contain one navigation table")
    start = text.index(_NAVIGATION_START)
    end = text.index(_NAVIGATION_END) + len(_NAVIGATION_END)
    if end < start:
        raise ValueError("metric navigation markers are reversed")
    return text[:start] + navigation_markdown() + text[end:]


def navigation_markdown() -> str:
    """Render primary metric locations without recounting shared viewer appearances.

    Returns
    -------
    str
        Marked Markdown table listing topic/view column totals and a unique-column sum.
    """
    counts = Counter(metric_primary_location(column) for column in METRIC_COLUMNS)
    lines = [
        _NAVIGATION_START,
        "| Topic | Primary columns in Totals | Primary columns By Recipient | "
        "Primary columns in single view |",
        "| --- | ---: | ---: | ---: |",
    ]
    for topic in METRIC_TOPICS:
        values = [
            str(counts[(topic.name, view)])
            for view in ("totals", "recipients", "single")
        ]
        if topic.paired:
            values[2] = "—"
        else:
            values[:2] = ["—", "—"]
        lines.append(f"| {topic.label} | {' | '.join(values)} |")
    lines.extend(
        (
            f"| **Total** | | | **{sum(counts.values()):,} unique measurements** |",
            _NAVIGATION_END,
        )
    )
    return "\n".join(lines)


_IDENTITY_MEANINGS = {
    "run_id": (
        "The run ID. Use it with phase, pass_id and episode_id to identify a row."
    ),
    "phase": (
        "The purpose you gave this run: training, validation, "
        "evaluation, or tournament."
    ),
    "pass_id": "The pass ID you chose within this run and phase.",
    "episode_id": (
        "The episode number within this pass, chosen before it runs. "
        "Episode numbers start at 1."
    ),
    "seed_id": (
        "The ID for this episode's stream of random numbers. "
        "It does not change with batch size or the order episodes finish."
    ),
    "map_id": ("The map ID. The map layout is stored in run_details.json."),
    "config_id": (
        "The SHA-256 fingerprint of the episode settings. "
        "The full settings are stored in run_details.json."
    ),
    "team_a_policy": (
        "Team A's recorded policy name. The pass details identify which "
        "parameters and checkpoint were used."
    ),
    "team_b_policy": (
        "Team B's recorded policy name. The pass details identify which "
        "parameters and checkpoint were used."
    ),
    "checkpoint_id": (
        "The saved checkpoint ID for this pass. Blank if the policy was still "
        "changing during training or no checkpoint was named."
    ),
}


def dictionary_csv() -> str:
    """Render the complete metric dictionary in the recorded CSV column order.

    Returns
    -------
    str
        CSV text with identity-field meanings, metric guidance, locations and indexes.
        Values are formatted from the shared catalog; no numerical metrics are run.
    """
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
            "family",
            "subject_role",
            "recipient_role",
            "numerator",
            "denominator",
            "gui_groups",
            "primary_topic",
            "primary_view",
            "full_run_column_number",
            "replay_column_number",
            "gui_text_by_topic",
        )
    )
    from marl_battlegrounds.evaluation.analysis import REPLAY_IDENTITY_COLUMNS

    replay_positions = {
        name: index for index, name in enumerate(REPLAY_IDENTITY_COLUMNS, 1)
    }
    for index, name in enumerate(IDENTITY_COLUMNS, 1):
        agent = name.startswith("agent_")
        active = name.endswith("_active")
        subject = name.split("_")[1] if agent else ""
        meaning = (
            "1 if this agent slot takes part in the episode; 0 if it does not."
            if agent and active
            else (
                "The class ID for this agent slot: Neutral=0, Mage=1, "
                "Warrior=2, Hunter=3, Rogue=4, Priest=5. "
                "Check its active column to see whether it takes part."
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
                "Context dependent",
                "",
                "",
                "episode_identity",
                "identity",
                "",
                "",
                "",
                "Episode Details",
                "Episode Details",
                "",
                index,
                replay_positions.get(name, ""),
                "",
            )
        )
    rosters = tuple((class_id,) * 10 for class_id in range(1, 6)) + tuple(
        tuple((slot + offset) % 5 + 1 for slot in range(10)) for offset in range(5)
    )
    for index, column in enumerate(METRIC_COLUMNS, 1):
        # Slot columns are class-agnostic. List possible views, whose source
        # membership is resolved from the actual recorded roster at inspection.
        views = set()
        topic_text: dict[str, dict[str, str]] = {}
        for roster in rosters:
            for topic, view in metric_locations(column, roster):
                views.add((topic, view))
                text = metric_topic_text(column, topic, roster)
                if text:
                    topic_text[METRIC_TOPICS_BY_NAME[topic].label] = text
        primary_topic, primary_view = metric_primary_location(column)
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
                metric_guidance(column)[1],
                column.required_class_id or "",
                ""
                if column.status_channel is None
                else STATUS_LABELS[column.status_channel],
                column.family,
                column.subject_role,
                column.recipient_role or "",
                column.numerator or "",
                column.denominator or "",
                "; ".join(
                    topic.label
                    + (f": {METRIC_VIEW_LABELS[view]}" if topic.paired else "")
                    for topic in METRIC_TOPICS
                    for view in ("totals", "recipients", "single")
                    if (topic.name, view) in views
                ),
                METRIC_TOPICS_BY_NAME[primary_topic].label,
                METRIC_VIEW_LABELS[primary_view],
                len(IDENTITY_COLUMNS) + index,
                len(REPLAY_IDENTITY_COLUMNS) + index,
                json.dumps(topic_text, ensure_ascii=False) if topic_text else "",
            )
        )
    return output.getvalue()


def main() -> int:
    """Write or check the metric dictionary and the specification's family summary.

    ``--check`` compares existing text without writing. Normal mode updates the
    requested CSV and changes the specification only when its summary differs.

    Returns
    -------
    int
        Zero when the requested write or comparison succeeds.

    Raises
    ------
    SystemExit
        If arguments are invalid, help is requested, or checked text
        is stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/evaluation/metric_columns.csv")
    )
    args = parser.parse_args()
    expected = dictionary_csv()
    specification = _SPECIFICATION.read_text()
    updated_specification = specification_with_summary(specification)
    if args.check:
        if args.output.read_text() != expected:
            parser.error("metric dictionary is stale; rerun without --check")
        if specification != updated_specification:
            parser.error("metric family summary is stale; rerun without --check")
    else:
        args.output.write_text(expected)
        if specification != updated_specification:
            _SPECIFICATION.write_text(updated_specification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
