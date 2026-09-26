"""Check the population selection command, lazy export and complete demo wiring.

The CLI calls the same public selection owner once, preserving paths and errors.
Fresh processes prove package import and help load no numerical modules or files.
The example binds its rule before games, uses the returned member references in
order, and starts a separate test field only after a complete frozen decision.
These forwarding tests do not fit ratings, run games or claim learning quality.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli


@pytest.mark.parametrize("declaration", [None, "rules/selection.json"])
@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_command_forwards_to_public_owner_once(
    declaration: str | None,
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    decision = {
        "status": status,
        **({"population_id": "frozen"} if status == "complete" else {}),
    }

    def select(**arguments: object) -> dict[str, str]:
        calls.append(arguments)
        return decision

    monkeypatch.setitem(marl_bgs.__dict__, "select_initial_population", select)
    arguments = ["select-population", "saved/run", "--output-dir", "new/decision"]
    if declaration is not None:
        arguments.extend(["--declaration", declaration])
    assert _cli.main(arguments) == (0 if status == "complete" else 1)
    assert calls == [
        {
            "result": "saved/run",
            "output_dir": "new/decision",
            "declaration": declaration,
        }
    ]
    assert json.loads(capsys.readouterr().out) == decision


def test_command_reports_owner_errors_without_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def select(**arguments: object) -> dict[str, object]:
        calls.append(arguments)
        raise ValueError("Selection rule was not saved before games")

    monkeypatch.setitem(marl_bgs.__dict__, "select_initial_population", select)
    assert _cli.main(["select-population", "run", "--output-dir", "decision"]) == 1
    assert len(calls) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Selection rule was not saved before games" in output.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["select-population"],
        ["select-population", "run"],
        ["select-population", "--output-dir", "decision"],
    ],
)
def test_command_requires_result_and_output(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _cli.main(arguments)
    assert error.value.code == 2


def test_package_import_and_command_help_have_no_runtime_effects(
    tmp_path: Path,
) -> None:
    program = """
import sys
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli
assert 'select_initial_population' in marl_bgs.__all__
assert 'select_initial_population' not in marl_bgs.__dict__
try:
    _cli.main(['select-population', '--help'])
except SystemExit as error:
    assert error.code == 0
blocked = ('jax', 'numpy', 'scipy', 'flax', 'marl_battlegrounds.evaluation')
assert not [name for name in sys.modules
            if any(name == item or name.startswith(item + '.') for item in blocked)]
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--declaration" in result.stdout
    assert not list(tmp_path.iterdir())


def test_lazy_export_is_the_shared_selection_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation.population_selection import (
        select_initial_population,
    )

    monkeypatch.delitem(marl_bgs.__dict__, "select_initial_population", raising=False)
    assert marl_bgs.select_initial_population is select_initial_population


@pytest.mark.parametrize("complete", [False, True])
def test_example_declares_before_games_and_uses_frozen_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    from examples import population_selection

    events: list[tuple[str, object]] = []
    result = SimpleNamespace(status="complete", run_dir=tmp_path / "selection")
    members = [{"controller": {"reference": name}} for name in ["tdm-beta", "random"]]

    def tournament(**arguments: object) -> SimpleNamespace:
        events.append(("run", arguments))
        return result

    def select(value: object, **arguments: object) -> dict[str, object]:
        assert value is result
        events.append(("select", arguments))
        if not complete:
            return {"status": "incomplete", "members": []}
        return {"status": "complete", "members": members, "population_id": "frozen"}

    monkeypatch.setitem(marl_bgs.__dict__, "run_tournament", tournament)
    monkeypatch.setitem(marl_bgs.__dict__, "select_initial_population", select)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert population_selection.main(["--output-dir", str(tmp_path)]) == (
        0 if complete else 1
    )
    config = {
        "entrants": ["random", "tdm-alpha", "tdm-beta"],
        "maps": [42, 43, 44, 45, 46],
        "games_per_opponent": 10,
        "max_steps": 2,
        "seed": 17,
        "selection": {
            "size": 2,
            "entrant_order": ["random", "tdm-alpha", "tdm-beta"],
            "failure_policy": "require-complete-field",
        },
    }
    expected = [
        (
            "run",
            {
                "config": config,
                "output_dir": tmp_path / "selection",
                "num_envs": 2,
                "chunk_size": 2,
            },
        ),
        ("select", {"output_dir": tmp_path / "decision"}),
    ]
    if complete:
        expected.append(
            (
                "run",
                {
                    "config": {
                        "entrants": ["tdm-beta", "random"],
                        "maps": [47, 48, 49, 50, 51],
                        "games_per_opponent": 10,
                        "max_steps": 2,
                        "seed": 29,
                        "selection": {
                            "stage": "test",
                            "population": str(
                                tmp_path / "decision" / "population.json"
                            ),
                        },
                    },
                    "output_dir": tmp_path / "test",
                    "num_envs": 2,
                    "chunk_size": 2,
                },
            )
        )
    assert events == expected


def test_example_help_does_not_load_runtime_or_choose_backend(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "population_selection.py"
    program = f"""
import os
import runpy
import sys
previous = os.environ.get('JAX_PLATFORMS')
sys.argv = [{str(path)!r}, '--help']
try:
    runpy.run_path({str(path)!r}, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert os.environ.get('JAX_PLATFORMS') == previous
assert 'jax' not in sys.modules
assert 'numpy' not in sys.modules
assert 'marl_battlegrounds' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output-dir" in result.stdout
    assert not list(tmp_path.iterdir())
