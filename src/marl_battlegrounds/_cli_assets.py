"""Connect asset commands to shared verification, publication and cleanup.

The dispatcher imports this module only for models commands. It owns prompts
and terminal text; the asset authority owns files, hashes and downloads. No
controller is loaded and no tournament is executed here.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any


def _confirm(
    namespace: argparse.Namespace, parser: argparse.ArgumentParser, prompt: str
) -> bool:
    """Ask once after preview, or require --yes when stdin is not interactive."""
    if namespace.yes:
        return True
    if not sys.stdin.isatty():
        parser.error("Use --yes to confirm this work without an interactive terminal")
    try:
        return input(prompt + " [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _next_output(path: Path) -> Path:
    """Suggest an unused report-config path without reserving or writing it."""
    candidate = path.with_name(path.stem + "-reports.json")
    index = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_name(f"{path.stem}-reports-{index}.json")
        index += 1
    return candidate


def _print_next(
    namespace: argparse.Namespace, prepared: dict[str, Any], path: Path | None
) -> None:
    """Print explicit preparation/execution routes using verified path hints."""
    config = prepared["config"]
    print(f"Snapshot ID: {config['snapshot_id']}")
    prefix = [sys.executable, "-m", "marl_battlegrounds"]
    supplied = path or (Path(namespace.config) if namespace.config else None)
    config_args = ["--config", str(supplied)] if supplied is not None else []
    command = "canonical" if config.get("release") is not None else "tournament"
    roles = set(namespace.roles)
    full_args = ["--metrics", "full"] if "full_report" in roles else []
    if config.get("record_sources") and "outcomes_priority" not in roles:
        print("Prepare The Separate Outcome Records Before Reuse")
        prepare = [
            *prefix,
            "models",
            "download",
            *config_args,
            "--roles",
            "outcomes_priority",
        ]
        if namespace.cache_dir:
            prepare.extend(["--cache-dir", str(namespace.cache_dir)])
        if path is not None:
            reports = _next_output(path)
            prepare.extend(["--output-config", str(reports)])
            config_args = ["--config", str(reports)]
        print("Prepare Priority Reports: " + shlex.join(prepare))
        print(
            "After Report Preparation: "
            + shlex.join([*prefix, command, *config_args, *full_args])
        )
    else:
        execution = [*prefix, command, *config_args, *full_args]
        print("Next Command: " + shlex.join(execution))
    print("Full reports and replays are separate asset choices; no run has started.")


def _download(namespace: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Inspect once, confirm missing bytes and publish verified paths if requested."""
    from marl_battlegrounds.evaluation._asset_commands import (
        _preflight_prepared_config,
    )
    from marl_battlegrounds.evaluation.tournament_assets import (
        _inspect_tournament_assets,  # pyright: ignore[reportPrivateUsage]
    )

    roles = namespace.roles
    allowed = {"model", "outcomes_priority", "full_report", "replay"}
    if len(set(roles)) != len(roles) or not set(roles) <= allowed:
        parser.error(
            "Choose unique roles from model,outcomes_priority,full_report,replay"
        )
    if namespace.dry_run and (namespace.yes or namespace.output_config):
        parser.error("--dry-run cannot be combined with --yes or --output-config")
    if namespace.cache_dir and not namespace.dry_run and not namespace.output_config:
        parser.error(
            "--cache-dir requires --output-config for the next offline command"
        )
    context = _inspect_tournament_assets(
        namespace.config,
        cache_dir=namespace.cache_dir,
        roles=roles,
    )
    preview = context.preview
    target = (
        _preflight_prepared_config(
            namespace.output_config,
            config=context.expected_config,
            input_config=namespace.config,
        )
        if namespace.output_config
        else None
    )
    print("Requested Roles: " + ", ".join(roles))
    print(f"Cache Directory: {preview['cache_dir']}")
    print(f"Missing Bytes: {preview['bytes_missing']}")
    for identifier in preview["missing"]:
        print(f"Missing Asset: {identifier}")
    if namespace.dry_run:
        print("Inspection Only; No Files Changed")
        return 0
    if preview["missing"] and not _confirm(
        namespace, parser, "Download the listed assets?"
    ):
        print("Cancelled; No Files Changed")
        return 0
    prepared = context.finish(download=bool(preview["missing"]), confirmed=True)
    output = target.write(prepared["config"]) if target is not None else None
    print(f"Downloaded Bytes: {prepared['bytes_downloaded']}")
    if output is not None:
        print(f"Prepared Configuration: {output}")
    _print_next(namespace, prepared, output)
    return 0


def _clean(namespace: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Preview explicit cache entries and remove them only after confirmation."""
    from marl_battlegrounds.evaluation._asset_commands import _inspect_cache_cleanup

    if namespace.dry_run and namespace.yes:
        parser.error("--dry-run cannot be combined with --yes")
    if any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in namespace.sha256
    ):
        parser.error("--sha256 requires a complete lowercase SHA256 digest")
    cleanup = _inspect_cache_cleanup(namespace.sha256, cache_dir=namespace.cache_dir)
    preview = cleanup.preview
    print(f"Cache Directory: {preview['cache_dir']}")
    print(f"Selected Bytes: {preview['bytes']}")
    for entry in preview["entries"]:
        status = (
            f"{entry['size_bytes']} bytes" if entry["present"] else "Already Absent"
        )
        print(f"{entry['sha256']}: {status}")
    print("Stop other writers/downloads for these entries before cleanup.")
    print(
        "Saved runs and snapshots may share these files "
        "and need preparation after removal."
    )
    if namespace.dry_run:
        print("Inspection Only; No Files Changed")
        return 0
    if any(entry["present"] for entry in preview["entries"]) and not _confirm(
        namespace, parser, "Remove the selected cache files?"
    ):
        print("Cancelled; No Files Changed")
        return 0
    result = cleanup.apply()
    for digest in result["deleted"]:
        print(f"Removed: {digest}")
    return 0


def run_models(namespace: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Dispatch parsed download/clean arguments; shared I/O errors reach main."""
    return (
        _download(namespace, parser)
        if namespace.operation == "download"
        else _clean(namespace, parser)
    )
