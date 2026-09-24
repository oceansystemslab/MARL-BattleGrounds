"""Check that training commands parse options and call the public Python owners.

Help remains independent of training backends. These dispatch checks do not
replace the runner's real collect/update/save/resume workflow tests. A QMIX
JSON config reaches the same public train function with its qmix settings,
and a PQN-VDN JSON config with its pqn settings and a 1600-step interval.
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


def test_train_cli_passes_a_qmix_config_to_the_shared_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": "qmix",
                "num_envs": 4,
                "total_env_steps": 192,
                "qmix": {"rollout_length": 8, "buffer_size": 64},
            }
        )
    )
    calls: list[runner.TrainConfig] = []

    def fake_train(
        configuration: runner.TrainConfig, **kwargs: object
    ) -> runner.TrainResult:
        del kwargs
        calls.append(configuration)
        return runner.TrainResult(
            tmp_path, tmp_path / "actor", None, 192, 3, "complete", ()
        )

    monkeypatch.setattr(runner, "train", fake_train)
    assert (
        _cli.main(["train", "--config", str(config), "--output-dir", str(tmp_path)])
        == 0
    )
    assert calls[0].method == "qmix" and calls[0].qmix is not None
    assert (calls[0].qmix.rollout_length, calls[0].qmix.buffer_size) == (8, 64)
    assert calls[0].checkpoint_interval_updates == 1600


def test_train_cli_passes_a_pqn_config_to_the_shared_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": "pqn_vdn",
                "num_envs": 4,
                "total_env_steps": 176,
                "pqn": {
                    "rollout_length": 4,
                    "memory_window": 2,
                    "epochs": 1,
                    "num_minibatches": 2,
                },
            }
        )
    )
    calls: list[runner.TrainConfig] = []

    def fake_train(
        configuration: runner.TrainConfig, **kwargs: object
    ) -> runner.TrainResult:
        del kwargs
        calls.append(configuration)
        return runner.TrainResult(
            tmp_path, tmp_path / "actor", None, 176, 20, "complete", ()
        )

    monkeypatch.setattr(runner, "train", fake_train)
    assert (
        _cli.main(["train", "--config", str(config), "--output-dir", str(tmp_path)])
        == 0
    )
    assert calls[0].method == "pqn_vdn" and calls[0].pqn is not None
    assert (calls[0].pqn.rollout_length, calls[0].pqn.memory_window) == (4, 2)
    assert calls[0].checkpoint_interval_updates == 1600
