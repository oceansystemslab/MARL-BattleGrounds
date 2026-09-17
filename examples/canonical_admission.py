"""Run an isolated, artificial maintainer admission and prepare a monthly release.

Install MARL-BGs, then run ``JAX_PLATFORMS=cpu python
examples/canonical_admission.py --directory /tmp/marlbg-admission-example``.
The directory must not exist. Keep canonical_fixture.py beside this script when
copying the example outside the checkout. The script creates synthetic incumbent
reports, executes 120 one-transition challenger games with full metrics, verifies
promotion and prepares a local monthly snapshot. It does not install a catalog
or qualify a real controller. CPU use here checks the workflow, not its speed.

The patched private clock and explicitly artificial rule approvals belong only
to this example. Real maintainers use actual time and separately approved rules.
The private maintainer helpers are separate from ordinary researcher calls.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

from canonical_fixture import build_record_bundle

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation import admission
from marl_battlegrounds.evaluation.tournament_config import canonical_json

# This example deliberately exercises private maintainer operations and clock.
# pyright: reportPrivateUsage=false


def _asset(path: Path, value: object, role: str) -> dict[str, Any]:
    """Write one immutable example JSON file and return its verified-asset fields.

    path is a new file in the example directory. value must be finite JSON data.
    role is registration or qualification. The returned mapping declares exact
    byte size and SHA-256 content identity; it has no download URL.
    """
    data = canonical_json(value)
    with path.open("xb") as stream:
        stream.write(data)
    return {
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "role": role,
        "path": str(path),
        "url": None,
    }


def run(directory: Path) -> None:
    """Create a complete fixture admission in a new directory and print its result.

    directory is created here and must not already exist. The example preserves
    every generated asset, saved game, local store and prepared release there.
    It supplies explicit fixture-only rules, including an incumbent tie order;
    these are examples of approval evidence, not official scientific decisions.

    Games use the shared evaluator, full metric recorder and existing rating
    calculation. Resume loads the complete saved result without another action
    or fit. Applying that same decision twice changes membership only once.
    Release preparation verifies all 66 surviving matchups before writing a
    snapshot. No publication operation is called.
    """
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    bundle = build_record_bundle(directory / "incumbents", full=True, max_steps=1)
    extra = build_record_bundle(
        directory / "challenger", entrants=13, maps=1, max_steps=1
    )
    participant = extra["config"]["participants"][-1]
    asset_ids = {participant["registration_asset"]}
    asset_ids.update(
        value["asset_id"]
        for key, value in participant["controller"]["content"].items()
        if key != "adapter_bindings" and value is not None
    )
    descriptor = {
        "participant": participant,
        "assets": {key: extra["config"]["assets"][key] for key in asset_ids},
    }
    rules = {
        "approval_id": "artificial-example-only",
        "fixture_only": True,
        "games_per_opponent": 10,
        "resource_limits": {"fixture_approved": True},
        "timing_definitions": {"fixture_approved": True},
        "scientific_eligibility": {"fixture_approved": True},
        "reproduction": {"fixture_approved": True},
        "public_test_feedback": {"fixture_approved": True},
        "episode_local_adaptation": {"fixture_approved": True},
        "revised_method_policy": None,
        "tied_incumbent_eviction": {
            "entrant_order": [
                item["entrant_id"] for item in bundle["config"]["participants"]
            ]
        },
    }
    store = directory / "maintainer-store"
    state = admission.initialize_admission_store(store, bundle["config"], rules=rules)
    submission: dict[str, Any] = {
        "submission_id": "artificial-example-submission",
        "controller_id": participant["controller_id"],
        "descriptor_asset": _asset(
            directory / "submission.json", descriptor, "registration"
        ),
        "received_at_utc": "2026-10-01T00:00:00+00:00",
        "complete_at_utc": "2026-10-28T00:00:00+00:00",
        "qualification_asset": _asset(
            directory / "qualification.json",
            {
                "controller_id": participant["controller_id"],
                "rules_approval_id": state["rules_approval_id"],
                "approved": True,
                "revises_controller_id": None,
            },
            "qualification",
        ),
    }
    release_at = "2026-11-01T00:00:00+00:00"
    with patch.object(
        admission, "_utc_now", return_value=datetime(2026, 10, 29, tzinfo=UTC)
    ):
        admission.register_submission(store, submission)
        attempt = admission.prepare_admission(
            store, submission["submission_id"], release_at=release_at
        )
        result = admission.execute_admission(
            store,
            attempt["attempt_id"],
            output_dir=directory / "runs",
            num_envs=32,
            chunk_size=1,
        )
        assert result.run_dir is not None
        resumed = admission.execute_admission(
            store,
            attempt["attempt_id"],
            resume_from=result.run_dir,
            num_envs=32,
            chunk_size=1,
        )
        assert resumed.metadata["executed_this_call"] == 0
        applied = admission.apply_admission(store, attempt["attempt_id"], resumed)
        assert (
            admission.apply_admission(store, attempt["attempt_id"], resumed) == applied
        )
    with patch.object(
        admission, "_utc_now", return_value=datetime(2026, 11, 1, tzinfo=UTC)
    ):
        prepared = admission.prepare_release(store, release_at)
        assert admission.prepare_release(store, release_at) == prepared
    assert not (store / "catalog.json").exists()
    print("Artificial Admission — Not An Official Release")
    print("Imported Package:", marl_bgs.__file__)
    print("Saved Run:", result.run_dir)
    print("Reused Games:", result.reused_games)
    print("New Full-Capture Games:", result.executed_games)
    print("New Games During Resume:", resumed.metadata["executed_this_call"])
    print("Admission Decision:", applied["decision"])
    print("Prepared Snapshot:", prepared["snapshot_id"])
    print("Verified Retained Matchups:", prepared["asset_evidence"]["matchups"])
    print("Submission Cutoff:", prepared["cutoff_at_utc"])
    print("Monthly Release Instant:", prepared["release_at_utc"])
    print("Local Store:", store)
    print("No Catalog Was Published")
    print(
        "After an interrupted attempt, call execute_admission with the same "
        "attempt and resume_from saved run. Apply only its complete result. "
        "Missing approvals or a passed release boundary must be resolved explicitly."
    )


def main() -> None:
    """Parse one new output directory and run the artificial admission workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, required=True, help="New directory for fixture files"
    )
    run(parser.parse_args().directory)


if __name__ == "__main__":
    main()
