"""Check package command forwarding, omission, trusted factories and real results.

Call spies prove that command parsing does not replace scientific authority.
Fresh processes check lazy help. Short CPU games compare Python/CLI outcomes,
full measurements and submitted replay actions under the same exact conditions.
Evaluation and tournament rule flags keep ordinary Python types. Entrant and
challenger references reach the tournament API unchanged; factories never run
while parsing. Evaluation and tournaments default to 128 environments; explicit
capacities pass through unchanged. Fresh-execution flags preserve explicit true
and false values; omission inherits saved settings. Canonical conditions remain
owned by its released field. The head-to-head preview prints the shared
accessor's counts and points, with exact IDs beside possibly repeated names.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli
from marl_battlegrounds.evaluation.results import EvaluationResult, TournamentResult


@pytest.mark.parametrize(
    "command",
    (
        (),
        ("evaluate",),
        ("train",),
        ("analyze-training",),
        ("canonical",),
        ("tournament",),
        ("models",),
        ("models", "download"),
        ("models", "clean"),
        ("replay",),
    ),
)
def test_help_has_no_runtime_imports_or_files(
    command: tuple[str, ...], tmp_path: Path
) -> None:
    script = f"""
import sys
from marl_battlegrounds._cli import main
try:
    main({[*command, "--help"]!r})
except SystemExit as error:
    assert error.code == 0
blocked = ('jax', 'numpy', 'matplotlib', 'marl_battlegrounds.evaluation',
           'marl_battlegrounds.viewer', 'marl_battlegrounds._cli_assets')
assert not [name for name in sys.modules
            if any(name == p or name.startswith(p + '.') for p in blocked)]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "command,expected",
    (
        (
            [
                "evaluate",
                "--system",
                "random",
                "--opponent",
                "tdm-alpha",
                "--episodes",
                "2",
            ],
            {
                "system": "random",
                "opponent": "tdm-alpha",
                "num_episodes": 2,
                "phase": "evaluation",
                "pass_id": "1",
            },
        ),
        (["canonical", "--resume-from", "old"], {}),
        (
            [
                "canonical",
                "--no-rerun-existing",
                "--metrics",
                "priority",
                "--replay-episodes",
                "",
            ],
            {"rerun_existing": False, "metrics": "priority", "replay_episodes": ()},
        ),
        (["tournament", "--config", "population.json"], {"config": "population.json"}),
        (["tournament", "--resume-from", "old"], {}),
    ),
)
def test_exact_forwarding_keeps_omitted_keys_absent(
    command: list[str],
    expected: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def receive(name: str, arguments: dict[str, Any]) -> None:
        received.append((name, arguments))

    monkeypatch.setattr(_cli, "_run_experiment", receive)
    assert _cli.main(command) == 0
    assert received == [
        (
            command[0],
            {
                **expected,
                "output_dir": None,
                "resume_from": "old" if "--resume-from" in command else None,
                "num_envs": 128,
                "chunk_size": 16,
            },
        )
    ]


@pytest.mark.parametrize("command", ("canonical", "tournament"))
@pytest.mark.parametrize(
    ("options", "expected"),
    (((), None), (("--rerun-existing",), True), (("--no-rerun-existing",), False)),
)
def test_rerun_option_reaches_the_same_python_owner(
    command: str,
    options: tuple[str, ...],
    expected: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def receive(**arguments: object) -> None:
        calls.append(arguments)

    def ignore(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(
        marl_bgs,
        "run_canonical_tournament" if command == "canonical" else "run_tournament",
        receive,
    )
    monkeypatch.setattr(_cli, "_print_result", ignore)
    assert _cli.main([command, "--resume-from", "saved", *options]) == 0
    assert len(calls) == 1
    if expected is None:
        assert "rerun_existing" not in calls[0]
    else:
        assert calls[0]["rerun_existing"] is expected
    assert calls[0]["resume_from"] == "saved"


@pytest.mark.parametrize("other_budget", (8, 10))
def test_tournament_references_rules_and_budget_aliases_reach_the_api(
    other_budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}

    def tournament(**values: object) -> None:
        received.update(values)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("The tournament API owns reference loading")

    monkeypatch.setattr(marl_bgs, "run_tournament", tournament)
    monkeypatch.setattr(_cli, "_load_method", forbidden)

    def ignore(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(_cli, "_print_result", ignore)
    assert (
        _cli.main(
            [
                "tournament",
                "--entrants",
                "random, research:make, /saved/actor",
                "--challenger",
                "challenger:make",
                "--games-per-opponent",
                "8",
                "--episodes-per-pair",
                str(other_budget),
                "--maps",
                "47,48",
                "--seed",
                "7",
                "--score-threshold",
                "5",
                "--max-steps",
                "20",
                "--red-zone-depth",
                "6",
                "--num-envs",
                "3",
                "--chunk-size",
                "2",
            ]
        )
        == 0
    )
    assert received == {
        "policies": ("random", "research:make", "/saved/actor"),
        "challenger": "challenger:make",
        "games_per_opponent": 8,
        "episodes_per_pair": other_budget,
        "maps": (47, 48),
        "seed": 7,
        "score_threshold": 5,
        "max_steps": 20,
        "red_zone_depth": 6.0,
        "num_envs": 3,
        "chunk_size": 2,
        "output_dir": None,
        "resume_from": None,
    }
    assert type(received["red_zone_depth"]) is float


def test_tournament_config_challenger_uses_the_same_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def tournament(**values: object) -> None:
        received.update(values)

    monkeypatch.setattr(marl_bgs, "run_tournament", tournament)

    def ignore(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(_cli, "_print_result", ignore)
    assert (
        _cli.main(
            ["tournament", "--config", "field.json", "--challenger", "research:make"]
        )
        == 0
    )
    assert received == {
        "config": "field.json",
        "challenger": "research:make",
        "output_dir": None,
        "resume_from": None,
        "num_envs": 128,
        "chunk_size": 16,
    }


def test_tournament_api_budget_conflict_is_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def tournament(**values: object) -> None:
        calls.append(values)
        raise ValueError("games_per_opponent and episodes_per_pair must agree")

    monkeypatch.setattr(marl_bgs, "run_tournament", tournament)
    assert (
        _cli.main(
            [
                "tournament",
                "--entrants",
                "random,tdm-alpha",
                "--games-per-opponent",
                "8",
                "--episodes-per-pair",
                "10",
            ]
        )
        == 1
    )
    assert len(calls) == 1
    assert calls[0]["games_per_opponent"] == 8
    assert calls[0]["episodes_per_pair"] == 10
    assert "must agree" in capsys.readouterr().err


def test_ordered_rosters_and_all_explicit_evaluation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def receive(_: str, values: dict[str, Any]) -> None:
        received.update(values)

    monkeypatch.setattr(_cli, "_run_experiment", receive)
    assert (
        _cli.main(
            [
                "evaluate",
                "--system",
                "research:make",
                "--opponent",
                "tdm-alpha",
                "--episodes",
                "20",
                "--maps",
                "47, 48",
                "--system-roster",
                "mage,priest",
                "--opponent-roster",
                "rogue,rogue,priest",
                "--spawn-mode",
                "paired",
                "--seed",
                "0",
                "--metrics",
                "full",
                "--save-replays",
                "2",
                "--full-metrics-episodes",
                "1,2",
                "--replay-episodes",
                "1,2",
                "--score-threshold",
                "20",
                "--max-steps",
                "300",
                "--phase",
                "custom",
                "--pass-id",
                "checkpoint",
                "--num-envs",
                "3",
                "--chunk-size",
                "7",
            ]
        )
        == 0
    )
    assert received == {
        "system": "research:make",
        "opponent": "tdm-alpha",
        "num_episodes": 20,
        "maps": (47, 48),
        "system_roster": ("mage", "priest"),
        "opponent_roster": ("rogue", "rogue", "priest"),
        "spawn_mode": "paired",
        "seed": 0,
        "metrics": "full",
        "save_replays": 2,
        "full_metrics_episodes": (1, 2),
        "replay_episodes": (1, 2),
        "score_threshold": 20,
        "max_steps": 300,
        "phase": "custom",
        "pass_id": "checkpoint",
        "num_envs": 3,
        "chunk_size": 7,
        "output_dir": None,
        "resume_from": None,
    }


@pytest.mark.parametrize(
    "arguments",
    (
        [],
        ["tournament"],
        ["canonical", "--maps", "47"],
        [
            "evaluate",
            "--system",
            "research:make",
            "--opponent",
            "random",
            "--episodes",
            "2",
            "--maps",
            "47,,48",
        ],
        [
            "evaluate",
            "--system",
            "research:make",
            "--opponent",
            "random",
            "--episodes",
            "2",
            "--system-roster",
            "",
        ],
        [
            "evaluate",
            "--system",
            "research:make",
            "--opponent",
            "random",
            "--episodes",
            "2",
            "--output-dir",
            "new",
            "--resume-from",
            "old",
        ],
        ["canonical", "--met", "full"],
        *(
            [
                "evaluate",
                "--system",
                "research:make",
                "--opponent",
                "random",
                "--episodes",
                "2",
                "--red-zone-depth",
                text,
            ]
            for text in ("abc", "nan", "inf", "1e3", "1_0", "1.2.3", "", "+", "\u0665")
        ),
        ["tournament", "--entrants", "random,,tdm-alpha"],
        ["tournament", "--entrants", "random,tdm-alpha", "--config", "field.json"],
        ["tournament", "--entrants", "random,tdm-alpha", "--red-zone-depth", "nan"],
        ["canonical", "--red-zone-depth", "5.0"],
    ),
)
def test_usage_errors_do_not_load_methods(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Method must not load on parser failure")

    monkeypatch.setattr(_cli, "_load_method", forbidden)
    with pytest.raises(SystemExit) as error:
        _cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("text", "depth"),
    (("6", 6.0), ("6.0", 6.0), ("0", 0.0), ("+2.5", 2.5), ("-0.5", -0.5), (".5", 0.5)),
)
def test_red_zone_depth_reaches_evaluate_as_a_float(
    text: str, depth: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}

    def receive(_: str, values: dict[str, Any]) -> None:
        received.update(values)

    monkeypatch.setattr(_cli, "_run_experiment", receive)
    command = ["evaluate", "--system", "random", "--opponent", "random"]
    assert _cli.main([*command, "--episodes", "2", "--red-zone-depth", text]) == 0
    assert type(received["red_zone_depth"]) is float
    assert received["red_zone_depth"] == depth


def test_factory_runs_once_per_occurrence_without_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = marl_bgs.policy("random")
    calls: list[int] = []
    module = types.ModuleType("packet8_factory")

    def factory() -> object:
        calls.append(1)
        return method

    module.factory = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    received: list[dict[str, Any]] = []

    def evaluate(**kwargs: object) -> None:
        received.append(kwargs)

    def ignore(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(marl_bgs, "evaluate", evaluate)
    monkeypatch.setattr(_cli, "_print_result", ignore)
    assert (
        _cli.main(
            [
                "evaluate",
                "--system",
                "packet8_factory:factory",
                "--opponent",
                "packet8_factory:factory",
                "--episodes",
                "2",
            ]
        )
        == 0
    )
    assert calls == [1, 1]
    assert received[0]["system"] is received[0]["opponent"] is method


@pytest.mark.parametrize(
    "reference",
    (
        "builtins:len",
        "builtins:None",
        "builtins:len.bad",
        "builtins:len:extra",
        "missing_packet8_module:make",
    ),
)
def test_factory_errors_keep_original_cause(reference: str) -> None:
    with pytest.raises(_cli._FactoryError) as error:
        _cli._load_method(reference)
    assert error.value.__cause__ is not None


def test_wrong_factory_type_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("packet8_bad_factory")
    module.make = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="System or Policy"):
        _cli._load_method("packet8_bad_factory:make")


@pytest.mark.parametrize(
    "options",
    (
        ["--static"],
        ["--static", "--frame-index", "0", "--port", "0"],
    ),
)
def test_replay_option_matrix_errors_are_usage_errors(options: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _cli.main(["replay", "missing-replay.json", *options])
    assert error.value.code == 2


def test_unexpected_programming_error_keeps_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Programming fault")

    monkeypatch.setattr(_cli, "_run_experiment", fail)
    with pytest.raises(AssertionError, match="Programming fault"):
        _cli.main(["canonical"])


def test_preview_reads_one_chunk_and_closes_iterator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state: list[str] = []

    def rows(name: str, *, rows: int) -> Iterator[dict[str, list[object]]]:
        assert name == "episodes" and rows == 32
        try:
            state.append("read")
            yield {
                "episode_id": [1],
                "episode_length": [2],
                "team_a_score": [0],
                "team_b_score": [0],
                "system_game_score": [None],
            }
            raise AssertionError("Preview requested another chunk")
        finally:
            state.append("closed")

    fake = types.SimpleNamespace(
        status="complete",
        run_dir=None,
        metadata={},
        completed_episode_ids=(1,),
        replays=(),
        replay_paths=(),
        iter_table=rows,
        head_to_head=lambda: {
            "system_id": ["a"],
            "system_name": ["Same"],
            "opponent_id": ["b"],
            "opponent_name": ["Same"],
            "games": [3],
            "wins": [1],
            "draws": [1],
            "losses": [1],
            "mean_points_for": [5.0],
            "mean_points_against": [2.0],
            "mean_point_margin": [3.0],
        },
    )
    _cli._print_result(cast(EvaluationResult, fake), tournament=False)
    assert state == ["read", "closed"]
    output = capsys.readouterr().out
    assert "Results Were Not Saved" in output
    assert "Unavailable" in output and "Up To 32 Rows" in output
    assert "Head To Head — Up To 32 Rows" in output
    assert "a\tSame\tb\tSame\t3\t1\t1\t1\t5.0\t2.0\t3.0" in output


def test_cpu_evaluation_matches_python_and_preserves_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = marl_bgs.evaluate(
        "random",
        "tdm-alpha",
        num_episodes=2,
        maps=(47,),
        system_roster=("mage", "priest"),
        opponent_roster=("rogue", "rogue", "priest"),
        max_steps=2,
        num_envs=2,
        chunk_size=2,
        metrics="full",
        save_replays=2,
    )
    received: list[EvaluationResult] = []

    def receive(value: EvaluationResult, **_: object) -> None:
        received.append(value)

    monkeypatch.setattr(_cli, "_print_result", receive)
    assert (
        _cli.main(
            [
                "evaluate",
                "--system",
                "random",
                "--opponent",
                "tdm-alpha",
                "--episodes",
                "2",
                "--maps",
                "47",
                "--system-roster",
                "mage,priest",
                "--opponent-roster",
                "rogue,rogue,priest",
                "--max-steps",
                "2",
                "--num-envs",
                "2",
                "--chunk-size",
                "2",
                "--metrics",
                "full",
                "--save-replays",
                "2",
            ]
        )
        == 0
    )
    actual = received[0]
    assert actual.episodes == expected.episodes
    for name, values in expected.full_metrics.items():
        if np.asarray(values).dtype.kind in "fiu":
            np.testing.assert_array_equal(actual.full_metrics[name], values)
    assert len(actual.replays) == len(expected.replays) == 2
    for got, wanted in zip(actual.replays, expected.replays, strict=True):
        for got_frame, wanted_frame in zip(got.frames, wanted.frames, strict=True):
            assert got_frame.model_dump(exclude={"episode_id", "frame_id"}) == (
                wanted_frame.model_dump(exclude={"episode_id", "frame_id"})
            )
    assert not list(tmp_path.iterdir())


def test_custom_command_uses_existing_results_without_refitting_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from canonical_record_fixtures import build_record_bundle
    from marl_battlegrounds.evaluation import tournament_statistics

    bundle = build_record_bundle(tmp_path / "inputs", entrants=3, maps=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(bundle["config"]))
    received: list[Any] = []

    def receive(value: EvaluationResult, **_: object) -> None:
        received.append(value)

    monkeypatch.setattr(_cli, "_print_result", receive)
    assert (
        _cli.main(
            [
                "tournament",
                "--config",
                str(config),
                "--metrics",
                "none",
                "--output-dir",
                str(tmp_path / "saved"),
            ]
        )
        == 0
    )
    first = received[-1]
    assert len(first.matches) == 6

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Completed resume must not fit again")

    monkeypatch.setattr(tournament_statistics, "summarize_tournament", forbidden)
    assert _cli.main(["tournament", "--resume-from", str(first.run_dir)]) == 0
    assert received[-1].matches == first.matches
    assert (
        _cli.main(
            [
                "tournament",
                "--resume-from",
                str(first.run_dir),
                "--no-rerun-existing",
            ]
        )
        == 0
    )
    assert received[-1].matches == first.matches
    before = {
        path: path.read_bytes() for path in first.run_dir.rglob("*") if path.is_file()
    }
    completed_calls = len(received)
    assert (
        _cli.main(
            ["tournament", "--resume-from", str(first.run_dir), "--rerun-existing"]
        )
        == 1
    )
    assert "rerun_existing" in capsys.readouterr().err
    assert len(received) == completed_calls
    assert before == {
        path: path.read_bytes() for path in first.run_dir.rglob("*") if path.is_file()
    }


def test_canonical_command_reuses_twelve_and_resumes_saved_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_canonical_tournament import _pin

    from canonical_record_fixtures import build_record_bundle
    from marl_battlegrounds.evaluation import canonical, tournament_statistics
    from marl_battlegrounds.evaluation import tournament_config as configs

    bundle = build_record_bundle(tmp_path / "inputs")
    selected = _pin(monkeypatch, bundle["config"])
    results: list[Any] = []
    original = _cli._print_result

    def receive(
        result: EvaluationResult | TournamentResult, *, tournament: bool
    ) -> None:
        results.append(result)
        original(result, tournament=tournament)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Complete reuse must not load models or choose actions")

    monkeypatch.setattr(_cli, "_print_result", receive)
    monkeypatch.setattr(canonical, "_active_pair", forbidden)
    assert _cli.main(["canonical", "--output-dir", str(tmp_path / "saved")]) == 0
    first = results[-1]
    assert first.reused_games == 660 and first.executed_games == 0
    assert len(first.table("tournament_rankings")["rank"]) == 12
    assert "Tournament Rankings" in capsys.readouterr().out
    catalog = configs._installed_catalog()
    catalog["default"] = "f" * 64
    monkeypatch.setattr(configs, "_installed_catalog", lambda: catalog)
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", forbidden)
    assert _cli.main(["canonical", "--resume-from", str(first.run_dir)]) == 0
    assert results[-1].snapshot_id == selected["snapshot_id"]
    assert results[-1].matches == first.matches
    before = {p: p.read_bytes() for p in first.run_dir.rglob("*") if p.is_file()}
    assert (
        _cli.main(
            ["canonical", "--resume-from", str(first.run_dir), "--metrics", "none"]
        )
        == 1
    )
    assert before == {
        p: p.read_bytes() for p in first.run_dir.rglob("*") if p.is_file()
    }


def test_thirteen_entry_command_shows_complete_rankings_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from canonical_record_fixtures import build_record_bundle
    from marl_battlegrounds.evaluation import canonical

    bundle = build_record_bundle(tmp_path / "inputs", entrants=13, maps=1)
    config = tmp_path / "population.json"
    config.write_text(json.dumps(bundle["config"]))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Complete records must not load models")

    monkeypatch.setattr(canonical, "_active_pair", forbidden)
    assert _cli.main(["tournament", "--config", str(config), "--metrics", "none"]) == 0
    output = capsys.readouterr().out
    assert "Results Were Not Saved" in output
    assert "Executed Games: 0" in output
    for entry in bundle["config"]["participants"]:
        assert entry["name"] in output
    assert not (tmp_path / "runs").exists()
