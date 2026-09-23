"""Check that training commands parse options and call the public Python owners.

Help remains independent of training backends. These dispatch checks do not
replace the runner's real collect/update/save/resume workflow tests.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds import _cli
from marl_battlegrounds.training import runner


@pytest.mark.parametrize("method", (None, "mappo", "ippo", "ff_mappo", "ff_ippo"))
def test_train_cli_calls_shared_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str | None,
) -> None:
    config = tmp_path / "config.json"
    settings: dict[str, object] = {"num_envs": 4, "total_env_steps": 8}
    if method is not None:
        settings["method"] = method
    config.write_text(json.dumps(settings))
    calls: list[tuple[runner.TrainConfig, dict[str, Any]]] = []

    def fake_train(
        configuration: runner.TrainConfig, **kwargs: object
    ) -> runner.TrainResult:
        calls.append((configuration, kwargs))
        return runner.TrainResult(
            tmp_path, tmp_path / "actor", None, 8, 1, "complete", ()
        )

    monkeypatch.setattr(runner, "train", fake_train)
    assert (
        _cli.main(["train", "--config", str(config), "--output-dir", str(tmp_path)])
        == 0
    )
    assert calls[0][0].num_envs == 4
    assert calls[0][0].method == (method or "mappo")
    assert calls[0][1] == {"output_dir": str(tmp_path), "resume_from": None}


def test_train_cli_requires_config_for_new_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        _cli.main(["train", "--output-dir", str(tmp_path)])
    assert error.value.code == 2


def test_resume_cli_does_not_invent_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, Any]]] = []

    def fake_train(configuration: object, **kwargs: object) -> runner.TrainResult:
        calls.append((configuration, kwargs))
        return runner.TrainResult(
            tmp_path, tmp_path / "actor", None, 8, 1, "complete", ()
        )

    monkeypatch.setattr(runner, "train", fake_train)
    assert _cli.main(["train", "--resume-from", str(tmp_path)]) == 0
    assert calls == [(None, {"output_dir": None, "resume_from": str(tmp_path)})]
