"""Keep sequential tournament passes, game keys and replays with one writer.

Small CPU Policy and System runs compare saved results and captured actions with
the ordinary evaluator. Interrupted execution keeps published rows and replay
bytes through resume. Job argument and legacy dispatch checks remain in
test_tournament_legacy.py; no worker, batching coordinator or memory limit is used.
"""

# pyright: reportPrivateUsage=false

import csv
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    _evaluate_tournament_episodes,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    independent_policies,
    policy,
)
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.tournament_execution import (
    TournamentJob,
    execute_tournament_jobs,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _jobs() -> tuple[TournamentJob, ...]:
    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=3, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    return tuple(
        TournamentJob(
            group,
            "first",
            "second",
            tuple(EpisodeSpec(i, config, 12, seed_id=100 + i) for i in ids),
            1729,
            "evaluation",
            str(group),
            replay_episodes=ids,
        )
        for group, ids in enumerate(((1, 2), (3, 4)))
    )


@pytest.mark.parametrize("system_path", [False, True])
def test_sequential_passes_keep_rows_and_replay_actions_through_resume(
    tmp_path: Path, system_path: bool
) -> None:
    method = (
        independent_policies((policy("random"),)) if system_path else policy("random")
    )
    jobs = _jobs()

    @contextmanager
    def load(job: TournamentJob) -> Generator[tuple[System | Policy, System | Policy]]:
        yield method, method

    with RunWriter(tmp_path / "saved", phase="setup", pass_id="setup") as writer:
        stream = execute_tournament_jobs(
            jobs,
            load,
            writer=writer,
            run_id=writer.run_id,
            num_envs=2,
            metrics="priority",
            chunk_size=1,
        )
        assert next(stream)[0] is jobs[0]
        stream.close()
        root = writer.run_dir
        saved_replays = {
            path.name: path.read_bytes() for path in (root / "replays").glob("*.json")
        }
        assert len(saved_replays) == 2

    with RunWriter(resume_from=root, phase="setup", pass_id="setup") as writer:
        results = list(
            execute_tournament_jobs(
                jobs,
                load,
                writer=writer,
                run_id=writer.run_id,
                num_envs=2,
                metrics="priority",
                chunk_size=1,
            )
        )
        assert [job for job, _ in results] == list(jobs)
        assert all(
            (root / "replays" / name).read_bytes() == value
            for name, value in saved_replays.items()
        )
        with (root / "episodes.csv").open(newline="") as saved:
            rows = list(csv.DictReader(saved))
        assert len(rows) == 4
        assert {
            (row["pass_id"], int(row["episode_id"]), int(row["seed_id"]))
            for row in rows
        } == {
            (job.pass_id, episode.episode_id, episode.seed_id)
            for job in jobs
            for episode in job.episodes
        }
        replay_paths = list((root / "replays").glob("*.json"))
        assert len(replay_paths) == 4
        replays = [load_replay(path).replay for path in replay_paths]
        by_seed = {
            replay.header.context.seed_protocol.episode_seed: replay
            for replay in replays
        }
        assert set(by_seed) == {101, 102, 103, 104}
        for job, result in results:
            reference = _evaluate_tournament_episodes(
                method,
                method,
                job.episodes,
                seed=job.seed,
                num_envs=2,
                chunk_size=1,
                save_replays=2,
                run_id=writer.run_id,
                phase=job.phase,
                pass_id=job.pass_id,
            )
            assert result.episodes == (() if job is jobs[0] else reference.episodes)
            saved = {
                int(row["episode_id"]): row
                for row in rows
                if row["pass_id"] == job.pass_id
            }
            for episode in reference.episodes:
                row = saved[episode.episode_id]
                for column in (
                    "episode_id",
                    "seed_id",
                    "map_id",
                    "outcome",
                    "episode_length",
                    "team_a_score",
                    "team_b_score",
                ):
                    assert int(row[column]) == getattr(episode, column)
                assert row["config_id"] == episode.config_id
            for replay in reference.replays:
                actual = by_seed[replay.header.context.seed_protocol.episode_seed]
                assert actual.schema_version == 4
                assert len(actual.frames) == 4
                assert len(actual.transitions) == 3
                assert actual.frames == replay.frames
                assert actual.transitions == replay.transitions
