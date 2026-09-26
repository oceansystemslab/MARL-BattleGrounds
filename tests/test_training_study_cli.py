"""Check study command parity, exact paths and lazy public entry points.

Each command delegates once to the corresponding Python owner. New work needs
an explicit declaration; resume does not invent one. Help and status imports
must not initialize the numerical backend. No learner or GPU runs here.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds import _cli


@pytest.mark.parametrize(
    "action,owner", [("run", "run_study"), ("start", "start_study")]
)
@pytest.mark.parametrize("resume", [False, True])
def test_study_run_and_start_forward_exact_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    owner: str,
    resume: bool,
) -> None:
    from marl_battlegrounds.training import study

    calls: list[dict[str, Any]] = []

    def execute(**kwargs: object) -> dict[str, str]:
        calls.append(dict(kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(study, owner, execute)
    declaration = str(tmp_path / "study.json")
    output = str(tmp_path / "work")
    arguments = (
        ["--resume-from", output]
        if resume
        else ["--config", declaration, "--output-dir", output]
    )
    assert _cli.main(["study", action, *arguments]) == 0
    assert calls == [
        {
            "config": None if resume else declaration,
            "output_dir": None if resume else output,
            "resume_from": output if resume else None,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "action,owner", [("status", "study_status"), ("stop", "stop_study")]
)
def test_study_status_and_stop_use_the_named_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    owner: str,
) -> None:
    from marl_battlegrounds.training import study

    calls: list[str] = []

    def execute(study_dir: str) -> dict[str, str]:
        calls.append(study_dir)
        return {"status": "Saved"}

    monkeypatch.setattr(study, owner, execute)
    assert _cli.main(["study", action, str(tmp_path)]) == 0
    assert calls == [str(tmp_path)]
    assert json.loads(capsys.readouterr().out) == {"status": "Saved"}


@pytest.mark.parametrize("action", ["run", "start"])
def test_new_study_requires_a_declaration(tmp_path: Path, action: str) -> None:
    with pytest.raises(SystemExit) as error:
        _cli.main(["study", action, "--output-dir", str(tmp_path)])
    assert error.value.code == 2


def test_study_public_imports_and_help_do_not_load_numerical_backends() -> None:
    program = """
import sys
from marl_battlegrounds import training, _cli
for name in ('run_study', 'start_study', 'study_status', 'stop_study'):
    assert callable(getattr(training, name))
try:
    _cli.main(['study', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'jax' not in sys.modules
assert 'numpy' not in sys.modules
assert 'flax' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "foreground" in result.stdout


@pytest.mark.parametrize("status", ["incomplete", "time_limit", "stopped"])
def test_study_run_reports_unsuccessful_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    from marl_battlegrounds.training import study

    def execute(**kwargs: object) -> dict[str, str]:
        return {"status": status}

    monkeypatch.setattr(study, "run_study", execute)
    assert _cli.main(["study", "run", "--resume-from", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_study_example_uses_the_same_public_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    background: bool,
    resume: bool,
) -> None:
    from examples import training_study

    calls: list[tuple[object, dict[str, object]]] = []

    def execute(config: object = None, **kwargs: object) -> dict[str, str]:
        calls.append((config, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(
        training_study.training, "start_study" if background else "run_study", execute
    )
    arguments = ["--resume-from" if resume else "--output-dir", str(tmp_path)]
    if background:
        arguments.append("--background")
    assert training_study.main(arguments) == 0
    assert len(calls) == 1
    config, keywords = calls[0]
    if resume:
        assert config is None
    else:
        assert config == Path(training_study.__file__).with_suffix(".json")
    assert keywords == {
        "output_dir": None if resume else tmp_path,
        "resume_from": tmp_path if resume else None,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
