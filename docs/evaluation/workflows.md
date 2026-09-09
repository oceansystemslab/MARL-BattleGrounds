# Research workflows

Use your own trainer with `make`, one `evaluate` callable for validation and
evaluation, and `run_tournament` for paired cross-play. The
[runnable example](../../examples/evaluation.py) exercises these public calls
with the existing ALPHA and BETA controllers. All examples use complete standard
TDM episodes; changing the batch size does not change episode identities.
For evaluation and validation, `--episodes` is the total budget across maps,
cycled in order. Four episodes cover four of the five maps; use five or a
multiple of five when demonstrating coverage of the complete five-map split.

## Evaluate and inspect

```bash
JAX_PLATFORMS=cpu .venv/bin/python examples/evaluation.py evaluate \
  --episodes 4 --num-envs 4 --metrics full --save-replays 2 \
  --output-dir runs/evaluation
```

Use `JAX_PLATFORMS=cuda` with a CUDA-enabled JAX installation to select GPU.
Compilation affects the first call. The output prints the unique run directory
and named file paths. Open one saved replay with:

```bash
.venv/bin/python scripts/dev/replay_viewer.py --replay PATH_TO_REPLAY
```

The viewer offers **Up to Current Tick** and **Final Episode**, plus scalar CSV
and explicitly named episode-configuration exports. Historical replay files
remain readable. A saved replay does not automatically request full metrics
during execution.

Viewer CSV exports contain one row for the selected replay boundary. Their numeric
column names and order are the same scalar catalog used by completed run tables;
unavailable values remain empty. Viewer provenance also records the replay digest,
analysis source digest, scope, local `frame_index`, and recorded
`simulator_step_count`, plus all ten slots' classes, activity and policy identities.
Local frame zero may represent a later simulator tick in an authored or resumed
capture. Historical episode and policy identifiers remain unchanged. Completed
run tables instead use run/phase/pass, scheduled episode/seed/map/configuration,
policy and checkpoint identity columns. **Episode Details** downloads the recorded
context, configuration, completion and runtime metadata as JSON; it is separate
from the numeric CSV and contains no trajectory arrays.

Without `output_dir`, the same call creates no files:

```python
import pandas as pd
from marl_battlegrounds import evaluate

result = evaluate("tdm-alpha", "tdm-beta", num_episodes=100, metrics="full")
episodes = pd.DataFrame(result.full_metrics)
print(episodes["team_a_score"].mean())
```

Pandas is optional analysis software, not a simulator dependency. With file
output, load `pd.read_csv(result.paths["full_metrics"])` instead; persisted full
rows are not also retained in memory. Blank cells represent unavailable values;
real zeros remain zero. See the [column dictionary](metric_columns.csv).

## Choose metrics and replays independently

`metrics="priority"` is the default. `"none"` disables optional measurements;
`"full"` collects the complete scalar suite. Both selectors accept finite
Python integer iterables and use one-based episode IDs:

```python
from marl_battlegrounds import make

env = make(
    "tdm", num_envs=128, metrics="priority",
    full_metrics_episodes=range(1000, 50_001, 1000),
    replay_episodes=range(49_951, 50_001),
)
```

Selections are resolved once. No Python membership checks or file writes occur
inside a transition. Metrics use authoritative facts directly; they do not need
a replay file or a second simulator run.

## Train and record

`reset(key, config, episode_id=ids)` returns observations and wrapper state.
The state owns the dynamic configuration, exact masks and episode counters, so
`step(key, state, actions)` has the inputs Core needs. Configuration changes enter
through reset. Use `env.get_action_mask(state)` before sampling actions.

The unbatched environment composes with `jit`, `vmap` and `lax.scan`.
`num_envs=128` supplies native batching directly. Reset completed lanes with
`env.reset(key, next_config, episode_id=next_ids, state=state,
reset_mask=done.done)`. Other lanes continue unchanged. Reset is explicit;
terminal results retain the completed episode's ID.

Your trainer supplies actions, updates parameters and chooses when to log.
An optional `RunWriter` accepts `info` from one step or the entire `infos` tree
returned by a collected scan chunk. Call `writer.write(infos)` on the host;
passing only the last step would lose earlier completions and replay packets.
Initialize training recording with
`RunWriter("runs/training", phase="training", pass_id="1")` so episodes are
truthfully labelled as belonging to an evolving policy.
`writer.flush()` and context-manager exit establish durability and raise write
failures. No logger is required for training.

The default writer buffer holds 128 completed rows. For a large predeclared
priority-only evaluation, an existing control such as
`RunWriter("runs/evaluation", buffer_size=4096)` reduces repeated metadata writes.
Pass that writer to `evaluate(..., writer=writer)`. Larger buffers use more RAM
and leave more completions to repeat after abrupt interruption; full-metric rows
are substantially wider. Explicit `flush()` and normal closing establish the
same durability regardless of buffer size.

Replay capture has no horizon-sized buffer in environment state. Selected
packets returned by a scan still occupy that chunk's output memory; choose a
bounded logging chunk. The writer spools selected packets and publishes complete
replays. Empty replay selection removes the capture subtree entirely.

## Repeat validation in one run

```bash
JAX_PLATFORMS=cpu .venv/bin/python examples/evaluation.py validation \
  --episodes 4 --num-envs 4 --metrics full --output-dir runs/validation
```

This example reuses one writer for two passes on maps labelled `validation`.
Both passes append to the same tables, distinguished by `phase` and `pass_id`.
In your trainer, invoke these passes at the training episodes you choose. Wrap
the current frozen model with `Policy(name, apply, variables, initial_carry,
checkpoint=...)`; the callable receives exactly the authorized actor input and
mask. The two teams' checkpoint identities are recorded separately in pass
metadata. Episodes spanning training updates must be labelled as an evolving
policy, not falsely attributed to one checkpoint.

## Run a tournament

```bash
JAX_PLATFORMS=cpu .venv/bin/python examples/evaluation.py tournament \
  --episodes 20 --num-envs 8 --output-dir runs/tournament
```

Here `--episodes` is the budget per unordered policy pair, including every map
and both side assignments. Twenty episodes over five maps gives two independent
paired blocks per map: suitable for a workflow check, not a strong research
claim. The official Big 12 budget is 100 episodes per pair, or 6,600 total.

Ratings center on **1,200**. Qualified CPU fitting and 5,000 matched bootstrap
resamples run once after all matches. Missing matches or failed fits are errors.
Insufficient evidence produces an explicit unavailable interval. Full metrics
and replays are optional; mandatory match outcomes support ratings and the
matchup/per-map summaries. See the [statistical protocol](protocol.md#frozen-rating-and-uncertainty-contract).

## Resume and combine results

Each new `output_dir` call creates a unique child directory. Supplying an
existing run itself as a new output destination fails. Explicitly resume with
`resume_from=run_dir`, preserving policies, maps, seed and selections. A changed
batch or logging-chunk size does not change the scheduled matches. Durable
completed episodes are skipped; interrupted table suffixes are recovered before
continuing. Failures name the affected run directory and are raised and recorded
when storage permits.

Episode IDs are local to a pass. When combining runs, use
`run_id`, `phase`, `pass_id`, and `episode_id` together. Compute your own means,
medians, learning curves or other summaries from the scalar tables. The
tournament's specified population weighting is handled by its report.

## Performance qualification

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m scripts.dev.benchmark_evaluation \
  --sizes 64 128 256 512 1024 --repeats 5 \
  --output artifacts/m8-performance
```

This compares the full numerical pipeline on identical CPU/GPU facts, then
measures native GPU rollouts with metrics disabled, priority, full, sparse full
selection and selected replay capture. It reports compilation separately from
repeated warm timings, transfer/persistence costs, real transitions, padding,
memory and visible capacity failures. It does not silently substitute smaller
batches. These measurements do not establish learner VRAM or the one-day
competent-policy claim; those require actual training trials.

### Measured result: RTX 5090, 2026-09-08

All five requested native batch sizes completed without microbatching. This run
used an RTX 5090 (32,607 MiB reported), AMD Ryzen 9 9950X3D, JAX 0.10.1, CUDA 13
and driver 580.173.02, with JAX preallocation disabled. Each cohort cycled the five
held-out maps; every sixteenth lane used a repeated-class 3v2 roster and the
others canonical mirrored 5v5. ALPHA/BETA used 10% masked-valid action exploration.
Each episode had a distinct position/health/accepted-action trajectory.

The numerical comparison includes initialization, all 300 padded transition
updates and finalization of all 1,388 full scalars. It operates on identical
already-resident authoritative facts; simulation and file writing are timed
separately below. Integer counters and validity masks agreed exactly. Float32
results passed `rtol=3e-5, atol=0.002`; the two controlled-effect matrix products
use `HIGHEST` precision. Each timed execution was synchronized.

Warm times below are medians of five executions, with observed min–max in
parentheses. These are full-batch times, not the time to simulate an episode.

| Environments | GPU full metrics, ms | CPU full metrics, ms | GPU ms/episode | CPU ms/episode |
| --- | --- | --- | --- | --- |
| 64 | 9.325 (9.125–9.397) | 46.423 (44.660–48.392) | 0.1457 | 0.7254 |
| 128 | 10.479 (10.124–10.557) | 80.976 (79.759–82.226) | 0.0819 | 0.6326 |
| 256 | 10.426 (10.333–10.502) | 146.835 (145.220–147.697) | 0.0407 | 0.5736 |
| 512 | 13.321 (13.189–13.450) | 222.054 (218.465–230.412) | 0.0260 | 0.4337 |
| 1,024 | 14.490 (14.247–14.677) | 308.298 (304.874–309.226) | 0.0142 | 0.3011 |

Compilation includes tracing, lowering and compilation of the measured function.
The first execution excludes compilation. Environment/configuration construction,
reset, imports and initial fact generation are outside these timings; adding
the two columns does not give total command startup time.

| Environments | GPU compile, s | GPU first execution, ms | CPU compile, s | CPU first execution, ms |
| --- | --- | --- | --- | --- |
| 64 | 8.383 | 12.714 | 7.977 | 48.724 |
| 128 | 8.227 | 13.214 | 5.141 | 84.119 |
| 256 | 8.427 | 13.756 | 7.627 | 152.710 |
| 512 | 8.486 | 15.657 | 2.853 | 222.452 |
| 1,024 | 8.337 | 17.405 | 3.008 | 309.245 |

Native rollout timings include exploratory policy application, Core transitions,
metric collection and retained terminal results. A cohort executes 300 fixed
steps with completed lanes masked; it does not refill lanes asynchronously.
`Sparse full` selects every sixteenth episode, and `Replay` uses priority metrics
with four independently selected captures. All modes reached the same final
Core states and requested metric values as the frozen facts.

| Environments | None, s | Priority, s | Full, s | Sparse full, s | Replay capture, s | Full minus none, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 14.086 | 14.121 | 14.144 | 14.290 | 14.117 | 57.71 |
| 128 | 14.126 | 14.115 | 14.231 | 14.428 | 14.252 | 105.03 |
| 256 | 14.225 | 14.253 | 14.274 | 14.597 | 14.267 | 48.49 |
| 512 | 14.311 | 14.301 | 14.360 | 15.003 | 14.369 | 49.09 |
| 1,024 | 15.589 | 15.594 | 15.613 | 16.993 | 15.636 | 23.93 |

Small differences between modes can overlap execution noise; negative priority
increments do not establish a speedup. Sparse collection genuinely skips
unselected full work, but its selected-lane dispatch was slower than dense full
collection here (at 1,024 lanes, 1.40 seconds above metrics disabled). Selection
controls evidence cost and disk output; it does not guarantee faster execution.

For completeness, native startup and warm spread are recorded separately for
each mode. First execution again excludes compilation and environment setup.

| Environments | Mode | Compile, s | First execution, s | Warm min–max, s |
| --- | --- | --- | --- | --- |
| 64 | none | 5.237 | 14.105 | 14.075–14.091 |
| 64 | priority | 5.570 | 14.137 | 14.102–14.127 |
| 64 | full | 17.320 | 14.163 | 14.132–14.163 |
| 64 | sparse_full | 13.959 | 14.215 | 14.258–15.144 |
| 64 | replay | 6.603 | 14.119 | 14.102–14.132 |
| 128 | none | 5.258 | 14.149 | 14.109–14.129 |
| 128 | priority | 5.517 | 14.146 | 14.111–14.206 |
| 128 | full | 17.090 | 14.266 | 14.216–14.256 |
| 128 | sparse_full | 14.200 | 14.438 | 14.410–14.436 |
| 128 | replay | 6.579 | 14.292 | 14.246–14.303 |
| 256 | none | 5.661 | 14.244 | 14.218–14.278 |
| 256 | priority | 5.620 | 14.237 | 14.240–14.272 |
| 256 | full | 17.429 | 14.267 | 14.256–14.278 |
| 256 | sparse_full | 14.261 | 14.616 | 14.583–14.614 |
| 256 | replay | 6.722 | 14.303 | 14.239–14.275 |
| 512 | none | 5.388 | 14.334 | 14.291–14.344 |
| 512 | priority | 5.369 | 14.286 | 14.270–14.312 |
| 512 | full | 17.380 | 14.351 | 14.340–14.392 |
| 512 | sparse_full | 14.395 | 15.029 | 14.973–15.020 |
| 512 | replay | 6.546 | 14.356 | 14.355–14.384 |
| 1,024 | none | 5.344 | 15.581 | 15.579–15.612 |
| 1,024 | priority | 5.411 | 15.590 | 15.574–15.615 |
| 1,024 | full | 17.416 | 15.670 | 15.590–15.622 |
| 1,024 | sparse_full | 14.350 | 16.981 | 16.970–17.003 |
| 1,024 | replay | 6.471 | 15.616 | 15.627–15.657 |

Every scheduled episode completed. Reported padding is executed fixed-cohort
work after completion; it contributes no metrics or extra replay transitions.

| Environments/completed episodes | Real transitions | Padding transitions | Episode length min / median / max | Worker peak RAM, GiB | Sampled worker VRAM, MiB |
| --- | --- | --- | --- | --- | --- |
| 64 | 14,348 | 4,852 | 181 / 218.5 / 300 | 4.106 | 1,120 |
| 128 | 28,502 | 9,898 | 163 / 216 / 300 | 3.800 | 1,120 |
| 256 | 56,697 | 20,103 | 163 / 214.5 / 300 | 4.100 | 1,632 |
| 512 | 113,271 | 40,329 | 163 / 215.5 / 300 | 3.973 | 2,874 |
| 1,024 | 227,038 | 80,162 | 163 / 216 / 300 | 4.451 | 4,924 |

RAM is whole-worker maximum RSS. VRAM is the maximum observed worker allocation
from a separate one-second `nvidia-smi` sampler, so short-lived peaks can be
missed. Both include benchmark-only frozen histories, CPU copies, compilation
and allocator retention. The command also reports JAX allocator peaks and
compiler argument/output/temporary sizes. These measurements are **not learner
memory requirements**. Retaining L128 observations alone at B1024 accounts for
6.27 GiB; parameters, gradients, optimizer state, activations, recurrent state,
action masks and other chosen rollout outputs add to that. An actual learning
update and the 24-hour competence criterion still require M10/M12 trials.

Host transfer and persistence are measured once per batch, separately from the
five numerical repeats. CSV timing includes buffered writing and durability.
Replay publication includes host preparation, spooling, JSON creation and fsync.

| Environments | Metric transfer, ms | Full CSV write + fsync, ms | Capture transfer, ms | Four replay files: build + write + fsync, s |
| --- | --- | --- | --- | --- |
| 64 | 1.549 | 45.108 | 14.482 | 27.247 |
| 128 | 1.262 | 68.397 | 13.874 | 27.580 |
| 256 | 1.878 | 115.227 | 15.146 | 27.622 |
| 512 | 1.447 | 214.040 | 11.336 | 27.678 |
| 1,024 | 1.825 | 400.950 | 13.497 | 27.565 |

The same four episode identities were selected at every size: their replay files
totaled 78,946,828 bytes. The deliberately long 300-step capture chunk returned
129,898,800 bytes of packets. Production recording should use bounded logging
chunks; no horizon-sized replay array is stored inside environment state.
Replay files remain substantially more expensive than scalar metrics and are
optional. The CSV run directories, including metadata, occupied approximately
0.71 / 1.35 / 2.62 / 5.15 / 10.24 MB at the five sizes.

Raw `batch-N.json` files produced by the command retain all samples, episode
lengths, compiler/allocator memory and per-file source hashes. The qualified
source-hash-map SHA-256 was
`3f841cfe5e2c22a5504dd61e51252205cbc3b8d781001a75e2174714a55e5b83`;
all five rows used that source with no changes during measurement. The subsequent
cleanup retired unused historical full-metric producers, an accumulator and the
old executor, plus their exports. The measured JAX metric, environment, policy,
evaluator, capture, writer and benchmark files remained byte-identical. Historical
artifact/report compatibility and the final integration gates are checked after
that retirement; raw benchmark identities are preserved rather than rewritten.
For comparable sampled VRAM, run this in a separate terminal during the benchmark
and stop it after completion; match its process IDs to `worker_pid` in each row:

```bash
nvidia-smi --query-compute-apps=timestamp,pid,used_gpu_memory \
  --format=csv,noheader,nounits --loop-ms=1000 > /tmp/marl-gpu-memory.csv
```

### Check the implementation

With the development dependencies and Node.js 24 installed, these commands run
the approved scenario qualification and the complete Python/frontend checks:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m scripts.dev.qualify_tdm_scenarios \
  artifacts/tdm-qualification
scripts/dev/check.sh
scripts/dev/check_frontend.sh
```

For CUDA integration on a clean committed checkout:

```bash
scripts/dev/check_gpu.sh
```

Before making a commit, finish formatting, stage the complete intended candidate
and run `scripts/dev/check_before_commit.sh`. This runs both complete local gates
and verifies that the staged candidate stayed unchanged. It intentionally requires
a nonempty staged candidate; use the individual checks above on a clean checkout.
