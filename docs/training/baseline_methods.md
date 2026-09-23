# Baseline Development Methods

**Status, 22 September 2026:** every learning result in this document comes from
models trained on the old maps (maps 0 to 41 with revision 6 of Map 39) and,
where a section says so, from games against ALPHA version 2 and BETA version 4.
All of those results are superseded and historical only. Methods, rules and
declarations still describe how the studies were run.

For the current model population, exact checkpoint locations, result tables and
manuscript evidence links, open the
[development checkpoint index](checkpoint_population.md). Population V1 keeps
40 trained snapshots plus one shared untrained reference. Future population
training is a proposal; current opponent sampling remains unchanged.

This record explains how baseline settings are tested before manuscript claims.
Keep every declared case, failure and incomplete case in the saved results.
Report native task scores separately from training rewards and combat measures.
Passing a software check does not establish useful learning. Baselines need not
show coordinated tactics to be valid reference methods.

## PPO Software And Experiment Scope

The shared PPO workflow supports `mappo`, `ippo`, `ff_mappo` and `ff_ippo`.
Choose the method in `TrainConfig`; training, save/resume, validation, selection,
export, loading and evaluation keep the same public routes. Recurrent IPPO
includes the existing curriculum and shaping options. The first feedforward
recipes are plain. See the [training guide](README.md#ppo-method-choices) for
model inputs, memory and complete examples, and the
[source ledger](source_reuse.md#ppo-model-choices) for parameter counts and
deliberate donor changes.

The new methods retain today's ValueNorm, input-scale and spawn-frame defaults.
The running MAPPO search does not choose their settings. Fixed-input donor
comparisons and tiny integration runs check software; they do not establish
sample efficiency, useful tactics or a ranking between methods. Feedforward
and recurrent models have different parameter counts, so this comparison is
not a pure memory ablation. Earlier MAPPO learning results below retain their
original historical scope.

## Current MAPPO Standard

**Superseded, 22 September 2026.** The study results quoted in this section come
from models trained on the old maps: maps 0 to 41 with revision 6 of Map 39,
before its update. They are historical only. The decisions recorded here stay
as written.

**Decision, 22 September 2026:** the recurrent MAPPO baseline trains with the
spawn frame on (`PPOConfig.spawn_frame="left"`, the default since this date;
the value `"right"` is removed), and selection and validation move to strong
opponents.
Every world-frame model trained and selected against Random before this date,
including the twenty-million-step tuning study's 48 runs and 480 saved
checkpoints and the pinned-opponent runs, is superseded as a baseline: not
carried into the Big 12, the checkpoint population or future comparisons, and
its Random-proxy ranking is not used to choose settings. The evidence stays in
its packages, each marked with a `SUPERSEDED.md`. A new sweep on the
spawn-frame recipe with strong-opponent selection decides the settings; the
512/32 shape below is kept as the starting configuration. The sweep may pin a
strong training opponent with `TrainConfig(pinned_opponent=...)`. Such an
opponent is a familiar opponent for that run: results against it, and all
eight protected-scenario results when it is `tdm-alpha` or `tdm-beta`, are
reported as familiar-opponent results, and pinning does not change the
declared validation panel.

**Decision, 20 September 2026:** use **512 parallel environments and rollout
length 32** as the standard starting configuration for future recurrent MAPPO
development runs. Set `num_envs=512` and `PPOConfig(rollout_length=32)` explicitly.
This records the experiment choice; Python constructor defaults still retain
the earlier values described in the training guide.

This choice follows the completed twelve-case screen and its five-case follow-up.
All 17 runs completed their declared steps, updates and validation checks.
512/32 had the highest requested combined score, most kills and largest kill
advantage, while finishing sooner than the other leading configurations.

| Selected Run | Result |
| --- | --- |
| Training steps / learner updates | 6,094,848 / 372 |
| Warmed collection and update speed | 22,505 environment transitions per second |
| Warmed training / total run time | 4 minutes 30 seconds / 5 minutes 44 seconds |
| Average kills / deaths per validation game | 13.20 / 2.95 |
| Average kill difference | +10.25, versus -0.375 before training |
| Final wins / draws / losses | 7 / 33 / 0, against Random |
| Requested combined score | 8.625 |

The combined score is `0.5 * average_kill_difference + 0.5 * wins`, with wins
counted out of 40 games. The user chose this exploratory ranking after the first
screen; it was not a predeclared benchmark endpoint. Native task results remain
separate. 512/64 had one more win, with 8 wins, 32 draws and no losses.

The supporting recipe used training seed 19,044,601, canonical 5v5, K20/H300,
all 42 training maps, curriculum off, dense team kill/death shaping coefficient
0.01, input scale 0.01 and the existing 128-wide recurrent network. These results
support a practical MAPPO starting choice. They do not establish the best setting
across training seeds, curricula, opponents or other baseline methods. Each
method still needs its own evidence.

The complete sorted 17-case table, including deaths, kill difference, updates,
steps per second, times and the requested score, is saved locally in
`artifacts/m9-m10/packet-4/bt-extra-5min/reports/combined_results.md` and
`combined_results.csv` in that same directory. The CSV retains exact values,
source tables, run directories, actor paths and evaluation identities. Original
per-game records and learning curves remain in both experiment packages.

Together the cases contain 95,796,224 training transitions, 8,176 updates and
2,080 unique evaluation games. The shared initial 40-game evaluation is counted
once. Final checks use maps 42–46 and the same four seed pairs per map, with both
spawn arrangements. No protected test-map results enter this choice.

The five-case follow-up finished at 17:53:28 UTC on 20 September 2026 and took
33 minutes 37 seconds. Its warmed training times were roughly five minutes,
not equal timed stops. Warmed time and speed exclude the first block, which
contains compilation and one useful update. Total run time includes setup,
compilation, validation, saving and reports. Different total budgets also change
self-play history timing. The original common step checkpoint was 917,504;
the follow-up used 983,040. Keep these limits with the manuscript record.

## Leading B/T Settings Across Three Seeds

**Superseded, 22 September 2026.** This study has run on the old maps (maps 0 to
41 with revision 6 of Map 39); its results are historical only. Do not use them
as current evidence, for comparisons or to choose settings.

**Declared, 20 September 2026; awaiting the user's launch:** retain the existing
seed 19,044,601 and add only seeds 19,044,611 and 19,044,612 for B512/T32,
B512/T64, B768/T32 and B1024/T32. The user added B1024/T32 and clarified that
the original seed counts toward three seeds per configuration. This is eight
new runs, not a repetition of the earlier seventeen-run screen.

Use each configuration's original measured warmed collection/update throughput
to set an even number of full updates estimated to fit 300 seconds. Resolved
budgets are 6,750,208 steps / 412 updates for B512/T32; 6,684,672 / 204 for
B512/T64; 7,421,952 / 302 for B768/T32; and 7,798,784 / 238 for B1024/T32.
Each new seed uses the same budget for its configuration. These are approximate
time matches; record actual times. Do not run calibration again or impose an
automatic time limit. Run one GPU worker at a time and rotate configuration
order between the two seed blocks.

Keep dense team kill/death feedback at 0.01, input scale 0.01, canonical 5v5,
all 42 training maps, K20/H300, curriculum off and the earlier learner/opponent
settings. Keep training recording off and metrics `none`. Check initialization,
983,040 transitions, halfway and final against the same forty Random games on
validation maps 42–46, root seed 19,043,001 and paired spawn ends. Reuse initial
results only within a training seed after exact inference-identity verification.

The final report adds eight rows to a new copy of the original seventeen-row
table. Keep seed, source identities, steps, updates, throughput, measured times,
kills, deaths, kill difference, W/D/L and the same exploratory combined score.
Report each configuration's three-seed mean and range, and the two fresh seeds
separately. The original seed selected the candidates, so it is not independent
confirmation. Final checkpoints are the endpoints; no strongest-looking earlier
checkpoint is substituted. Native outcomes remain separate from the combined
score. This remains a short development check, not a universal B/T qualification.

The commands, declarations, pinned-source link and results live under
`artifacts/m9-m10/packet-4/bt-seeds-5min/`. The package is prepared in advance; the user
launches it and returns for analysis. No new learning evidence exists at this
declaration point.

## Curriculum And Dense Reward Comparison

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

On 20 September 2026, two runs compared the current curriculum with curriculum
off. Both used B512/T32, training seed 19,044,601, input scale 0.01, dense team
kill/death reward coefficient 0.01 and the same remaining learner settings.
Each completed 40,501,248 transitions and 2,472 updates. The two-run pipeline
took 1 hour 8 minutes 10 seconds. No training budget was extended.

Each run has ten trained validation checkpoints. All use the same forty Random
games on maps 42–46, root seed 19,043,001, paired spawn ends and K20/H300. The
initial forty-game result is shared, giving 840 unique evaluation games.

| Measurement | Curriculum + Reward Shaping | Reward Shaping Only |
| --- | ---: | ---: |
| Highest observed wins / draws / losses | 9 / 31 / 0 | 13 / 27 / 0 |
| Step at highest observed win rate | 12,156,928 | 28,360,704 |
| Highest observed win rate | 22.5% | 32.5% |
| Final wins / draws / losses | 7 / 33 / 0 | 7 / 33 / 0 |
| Final mean kills / deaths | 10.125 / 1.700 | 12.225 / 1.475 |
| Final mean kill difference | +8.425 | +10.750 |
| Collection and update time, including first compiled block | 31m 03s | 31m 52s |
| Whole run time | 33m 32s | 34m 31s |
| Warmed transitions per second | 21,790 | 21,233 |

Reward shaping only had the higher observed peak win rate and final combat
margin; final win rates tied. Both curves fluctuated. This is one training seed,
and peak checkpoints were chosen using these same forty games. These results
do not establish that curriculum is generally worse. All seventeen curriculum
stages received actual starts and transitions; every stage had at least
723,984 transitions. The 1v1 stage had only draws, while 2v2 did produce wins.

The RS-only run briefly shared the GPU with a 44-second check of the old
ten-hour final actor. Its recorded time includes that overlap. The old actor
scored 0 wins, 40 draws and 0 losses, with 5.6 kills and 0.5 deaths per game.
It used different reward, input, B/T, seed and budget settings. It is a historical
reference, not a controlled test of any single change.

A later check evaluated all twenty saved self-play actors from that old run,
at roughly 5% intervals, against the same forty Random games. It took 2 minutes
35 seconds for nineteen new checks and reuse of the verified final result.
Its highest observed win rate was 12.5%: 5 wins, 35 draws and no losses at
441,581,568 steps (45% of training; 4h 28m 58s into the original run). That
actor averaged 11.825 kills, 1.625 deaths and a +10.200 kill difference.
The original run therefore had a stronger earlier checkpoint than its final
actor. These post-training checks are not independent confirmation of a selected
model. Their raw records and original training times are retained under
`artifacts/m9-m10/packet-4/c-rs-10h-history-validation/`.

The complete checkpoint table, exact identities, time fields and comparison
plots are saved under
`artifacts/m9-m10/packet-4/curriculum-vs-rs-30min/reports/` as
`all_checkpoints.md`, `all_checkpoints.csv` and `checkpoint_comparison.png/.svg`.
The table ranks by win rate, then fewer losses and earlier training step. The
earlier exploratory combined score remains a separate column. The checkpoint
elapsed time precedes its own evaluation; final run totals include final outputs.

## Separate Runs On One GPU

**Workload note, 22 September 2026.** The measurements in this section used
training on the old maps (maps 0 to 41 with revision 6 of Map 39). They remain
valid measurements of the code's cost. They are not learning results.

On 20 September 2026, a bounded check measured one, two and four independent
B512/T32 MAPPO processes on the internal RTX 5090. Each process had four separate
physical CPU cores, preallocation off and JAX memory fraction 0.20. After twelve
warmup updates, a shared start began 24 timed updates per worker. One further
update finished the normal training workflow, followed by a 40-game Random check.
All seven workers completed training, saving and validation in 6 minutes 57 seconds
across the three tests. Matching seeds produced matching final actor identities.

| Concurrent Runs | Total Steps Per Second | Each Run Steps Per Second | Peak Board VRAM | Mean GPU Activity |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19,327 | 19,327 | 6.21 GiB | 97.3% |
| 2 | 19,435 | 9,724–9,729 | 11.21 GiB | 99.0% |
| 4 | 19,781 | 4,946–4,947 | 21.21 GiB | 99.0% |

Four fit, but total speed increased by only 2.35%, while each run took about
3.91 times as long. Use one process when time to the first result matters.
The check does not establish a hard maximum; higher counts were not tested.
The roughly 22,505 steps/second planning estimate remains an earlier measured
isolated-run rate, not a speed each simultaneous process will receive.

These rates include collection, PPO updates and ordinary host bookkeeping;
compilation, warmup, saving, validation and barrier waits are outside the timed
interval. Board memory includes driver/display use and was sampled about once
per second over the full workflow. Activity peaked at a sampled 99% in every
test. Per-process JAX pool peaks were 4.01 GiB. These are short workload-specific
measurements, with another seed and budget/history schedule than the learning
screen; they do not establish long-run stability or a statistically reliable
small speed advantage. The full record, exact CSV and raw results are under
`artifacts/m9-m10/packet-4/b512-t32-concurrency/`.

## Five-Minute MAPPO Configuration Screen

**Superseded, 22 September 2026.** This screen ran on the old maps (maps 0 to 41
with revision 6 of Map 39). Its learning results are historical only; its speed
checks remain valid cost measurements of that code.

The first screen asks which complete recurrent MAPPO configuration learns more
for the time spent. It is a small development experiment, with one training seed.
Five minutes is short, so wins are not expected or required. The main early
comparison is combat improvement against the same Random opponent: changes in
kills and deaths, measured against actual elapsed run time. All draws can still
contain useful combat learning. Combat improvement does not count as winning
the original task.
It does not choose the final published baseline or estimate training-seed
variation. An inconclusive screen ends in a review; it starts no longer run.

| Choice | Declaration |
| --- | --- |
| Parallel environments B | 32, 512 and 1,024 |
| Rollout length T | 16, 32, 64 and 128 |
| Cases | All twelve B/T pairs |
| Budget | Fixed steps estimated to take 300 seconds of collection and PPO updates |
| Training seed | 19,044,601, reused across cases |
| Calibration seed | 19,044,600, discarded before scientific training |
| Ordering seed | 19,044,602; Python `Random.shuffle` over ascending B then T |
| Task | Canonical 5v5, maps 0–41 uniformly eligible, K20/H300 |
| Curriculum | Off |
| Training reward | Native task reward plus 0.01 per team kill minus 0.01 per team death |
| Actor and critic | Existing recurrent MAPPO, width 128, input scale 0.01 |
| Remaining learner settings | Existing donor defaults, expanded in `declaration.json` |
| Optional output | Training `metrics="none"`; recording and replays off |
| Hardware | One worker on the explicitly identified internal RTX 5090 |

Each actor has separate memory. Teammates share actor weights. Dense team
kill/death shaping (`score_delta` in saved settings) is paid when the score
changes; it is not paid again just
because a lead remains. It changes the training objective. Evaluation keeps
the original task score and never adds shaping.

## Timing And Budget Rule

Each shape gets one cold collection/update call. Warmup then runs at least
300 further ticks per environment, with resets and historical opponents checked.
Five more completed blocks supply the median time. Each block includes the
existing required learner synchronization. The cold call is reported separately;
it combines compilation and execution and is not labelled pure compilation.

For median seconds per update `s`, freeze `U = 2 * floor(300 / (2 * s))` and
`N = U * B * T`. Reject nonfinite times or fewer than two updates. Even U gives
an exact halfway checkpoint. Save all twelve budgets and complete configurations
together before training starts. Rewards cannot influence this calculation.
Every scientific case starts fresh; calibration is never a training prefix.

Five minutes is an estimate, not a timer that silently cuts off a learner.
Report actual durations and achieved steps. The nominal total is about one hour
of warmed training, plus setup, compilation, validation, saving and reporting.
Calibration has a 20-minute limit. Before trials start, measured overhead,
projected work, a 25% margin and a three-minute shutdown reserve must fit the
original two-hour deadline. If they do not fit, stop without shrinking cases.

The numerical cutoff is 117 minutes after first launch. The remaining three
minutes cover cleanup and partial reporting. Resume keeps the original clock
and frozen budgets. Expired resume may reconcile reports but cannot train.
There are no automatic retries. Explicit resume may retry an unfinished
calibration before budgets exist; completed measurements remain fixed and all
attempts remain visible. A forced stop preserves the last durable checkpoint;
it does not promise a new checkpoint.

## Validation And Analysis

Use the existing Random diagnostic on validation maps 42–46, with four seed
pairs per map, both spawn arrangements and root seed 19,043,001: 40 games per
capture. Native score is win 1, draw 0.5, loss 0. Record W/D/L and per-map scores.
Report kill differential separately; it cannot turn a draw into a win.

Capture initialization, halfway and final. Also capture one common step count:
`A = min(1048576, 131072 * floor(min(N) / 131072))`. Omit A when it is zero;
sort and deduplicate coincident captures. Initialization may be reused only
when the actual actor inference identities match. Keep its original actor,
task and raw game references. Do not invent twelve independently evaluated
initial models. The maximum is 37 scientific diagnostic passes, or 1,480 games.
One separate 40-game calibration diagnostic measures evaluation cost and is
excluded from learning results.

The main curves show the change in average kills minus deaths per game since
before training, against each run's actual elapsed minutes. Show average kills
and deaths separately too, so a better difference is easy to understand. Keep
native task score and W/D/L alongside those results without making wins a
requirement for a useful early comparison. A flat win rate alone is not evidence
that learning failed. Do not automatically pick a winner from small differences.

Also show environment steps and measured training time with each attempt's cold
first block excluded. Include setup, evaluation and saving in whole-run costs.
Each plotted capture is timed just before its evaluation; final run totals also
include the last evaluation and reports. Pauses count toward wall time. The first
case pays for shared initialization evaluation; later cases reuse those same
results, so their elapsed times have that stated cost advantage.
Recover interrupted evaluation using its existing M8 games before collecting
more experience. A CPU worker handles fewer than 32 remaining games.

Resample complete seed pairs within each map. Reuse the same resampled blocks
across configurations and captures when comparing them. These intervals describe
game variation for the saved actors, not uncertainty across independent training
seeds. Shared seed numbers do not make trajectories identical across B/T shapes.
Different step budgets also move percentage-based self-play history captures.
The screen therefore compares complete configurations, not an isolated causal
effect of B or T. Random is a diagnostic opponent, not a competence panel.

On 20 September 2026, the first launch completed all twelve speed checks in
about 16 minutes 16 seconds, then stopped before any comparison trial. Its
forecast did not fit the original two-hour ceiling. This was a scheduling stop,
not a learning result. Preserve those measurements and that attempt in the
experiment history. A later user-approved change to the launch allowance must
be recorded separately; it does not change the five-minute case budgets.

## Saved Record And Later Decisions

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

The package owns `declaration.json`, `screen_package.json`, `launch_time.json`,
`budgets.json`, `study.json`, per-case configs, run records and attempt logs.
The source snapshot includes the actual reviewed public working files and their
hashes; its origin Git commit alone does not identify an uncommitted candidate.
Private milestone documents are excluded. The environment comes from the
unchanged lockfile. Package-local compilation caching follows the
[JAX cache contract](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
Cache reuse must be measured on actual collection and learner calls.

Reports contain one machine-readable `baseline_trials.csv`, `learning_curve.csv`,
`validation_cells.csv`, PNG/SVG plots and `run_summary.md`, with links to the
underlying M8 records. Failed and unstarted cases remain visible. Logs give UTC
timestamps, phase changes, progress, throughput, estimated remaining time and
total elapsed time. Reading status performs no JAX work.

Keep generated configs, caches, logs, checkpoints, plots and scratch evidence
under the chosen `artifacts/` output directory. The installable training package
contains reusable code only. Do not ship local experiment files as package data.

The completed screen's result and adopted starting configuration are recorded
under [Current MAPPO Standard](#current-mappo-standard). Review numerical
validity, actual budgets, costs and learning separately for each later run.
Any later search must declare sensible ranges, common evaluation, seed coverage,
selection rules and an affordable budget before it starts. Preserve this screen
in that history, including weak results. Final baseline claims require more than
this one-seed screen and must not use protected test results for tuning.

## Thirty-Minute Curriculum Comparison

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

**Declared, 20 September 2026; not yet run:** compare current curriculum plus
reward shaping against reward shaping only. Run sequentially on the internal
RTX 5090, with 512 environments, rollout length 32, shared seed 19,044,601 and
40,501,248 transitions (2,472 full updates) each. The user's planning rate of
22,505 steps/second gives about thirty minutes of training per case. Fixed steps
are equal; measured times may differ. No new calibration or timed stop is used.

The learner configs differ only in `curriculum`. Keep dense team kill/death
coefficient 0.01, input scale 0.01, existing PPO settings, K20/H300 and no training
replays or optional metrics. The curriculum uses the existing seventeen stages;
the other run uses canonical 5v5 and all forty-two training maps throughout.
Retain actual episode starts and transition exposure by curriculum stage.

Both runs check initialization and each resolved 10% step boundary on maps 42–46
against Random, using root 19,043,001 and four paired seeds per map: forty games.
Reuse the screen's initial check only after exact inference identity validation;
preserve original actor and task references. Twenty later checks add 800 games.
Use final actors for the comparison table, with the same requested score as the
screen: half average kill difference plus half win count out of forty. Also show
native W/D/L, kills, deaths, steps, updates, warmed speed and measured times.

This paired development check can show differences for one starting seed. It
cannot measure between-seed variation, prove a best curriculum or promise equal
trajectories. Different curricula change team sizes and actor decision counts;
report those alongside environment steps. Keep the existing K20 reachability
limits visible when interpreting early small-team stages. The complete recipe,
commands, logs and table live under
`artifacts/m9-m10/packet-4/curriculum-vs-rs-30min/`. The tools are prepared in advance; the
user launches them and returns for the result review.

## Twenty-Million-Step MAPPO Tuning Study

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

**Approved on 20 September 2026; all 48 runs completed on 21 September 2026.**
The completed repeat screen leaves B512/T32 first by the requested three-seed
combined score, narrowly ahead of B512/T64. On the two fresh seeds, B768/T32
leads. These results show seed sensitivity; they do not prove a universal winner.
The user chose **512 environments and 32-step rollouts** for this next study.
The original and fresh-seed tables remain under
`artifacts/m9-m10/packet-4/bt-seeds-5min/reports/`.

The new study crosses four choices: fixed K20 or threshold curriculum; entropy
coefficient 0.01 or 0.003; discount 0.99 or 0.999; and constant actor/critic
learning rates 0.00025 or 0.0001. Each of the sixteen settings gets fresh training
seeds 19,044,701, 19,044,702 and 19,044,703. Run all 48 sequentially on the
internal RTX 5090. A separate ordering seed, 19,044,704, shuffles settings within
each training-seed block. There is no new speed calibration, automatic deadline,
automatic retry or budget extension.

Each run gets **19,999,744 real environment transitions**: twenty million rounded
down to a whole batch of 512. This is 1,220 full updates and one final update
containing 11,264 transitions. Padding contributes neither experience nor loss.
Keep canonical 5v5, all training maps 0–41, H300, the recurrent 128-wide network,
input scale 0.01, existing PPO settings and current/historical self-play. Training
uses `metrics="none"` and no replays. Keep dense reward shaping on: native task
reward plus 0.01 per new team kill minus 0.01 per new team death. This is the
existing `score_delta` mode, not potential shaping and not a bonus for remaining
ahead. It changes the training objective; official evaluation scores stay native.

Threshold curriculum requests K1 for 10% of the step budget; K2 through K10 each
for 1/30; K12 and K15 each for 5%; and K20 for the final 50%. Resolve all shares
using the existing largest-remainder whole-batch rule. Apply a changed K only
when a game resets. Continuing games retain their original K. Record actual
starts, transitions and outcomes under each K; requested shares are not measured
exposure. The earlier team-size/map curriculum stays off in every study case.

Check initialization and each update-resolved 10% boundary against Random on
validation maps 42–46, canonical 5v5, K20/H300. Four seed pairs per map and both
spawn ends give forty games, using root 19,043,001. Share initialization only
within one training seed after exact inference and task verification. Preserve
the original actor and result references. After a run finishes, select most wins,
then fewest losses, then earliest positive training step. Freeze that selection
before evaluating it in **200 fresh games** with root 19,044,791 and twenty seed
pairs per map. Never change the selected model after seeing confirmation.

Rank settings by mean confirmation win rate across three training seeds, then
mean losses, earlier mean selected step and configuration ID. Report every seed,
final and selected checkpoints, native W/D/L, kills, deaths, kill difference,
steps, updates, measured times and throughput. Retain the earlier combined score
only on forty-game rows. Paired game uncertainty keeps both spawn ends together
and uses matching resamples across settings. Training-seed means and ranges are
reported separately. Confirmation helps choose settings, so it is development
evidence, not an untouched final benchmark test. Near-100% wins against Random
is an aim, not a promised result or proof of strength against learned opponents.

The package and commands live under
`artifacts/m9-m10/packet-4/mappo-tuning-20m/`. Its declaration, configs, source
hashes, lockfile environment, selection records and raw M8 games provide the
manuscript trail. Earlier longer runs do not replace fresh controls: their
percentage-based self-play history used different declared budgets. The package is checked first; the user launches it and returns for analysis. Longer follow-up
runs require a separate decision. The full repository gate and commit remain
deferred during this preparation.

### Completed Three-Seed Results

All 48 runs reached 19,999,744 transitions and 1,221 updates. All selected models
completed their 200 fresh confirmation games. The study took **13 hours,
54 minutes, 48 seconds**, including final reporting. The saved source and settings
were fixed before launch. No failed setting or seed was replaced.

The best mean confirmation win rate was **27.17% for c07**: fixed K20, entropy
0.003, gamma 0.999 and actor/critic learning rate 0.00025. Its three seeds scored
33%, 32% and 16.5%. **c03** scored 24.83% with a smaller observed seed range of
22.5-28.5%; it used entropy 0.01 with the other c07 settings. **c01**, the fixed
K20 reference at entropy 0.01, gamma 0.99 and learning rate 0.00025, scored 23.83%
and had the highest mean kill difference, +9.74 per game.

Across the full grid, fixed K20 averaged 20.10% wins and threshold curriculum
averaged 15.21%. Curriculum scored lower in six of eight otherwise matched
configuration pairs. All 13 threshold stages received real starts and experience
in every curriculum run. The early K1 stage produced decisive task outcomes in
about 95.5% of completed games, so this experiment did reduce early task-reward
sparsity. It did not establish a better curriculum. Some selected checkpoints,
especially c16's three 2-4M-step checkpoints, scored better on the routine checks
than their final models. The cause remains unresolved; noisy checks and selection
of observed peaks must be considered.

The [completed assessment](../../artifacts/m9-m10/packet-4/mappo-tuning-20m/reports/assessment.md)
contains the ranked table, costs, matched-setting comparisons, limits and links
to all 48 model identities, raw games, learning curves and threshold exposure.
The leading mean remains far below the near-100% Random win goal. Three seeds
support a shortlist, not a definitive universal winner. Confirmation was used
to choose configurations and remains development evidence. No longer experiment
was started during this assessment; the full repository gate and commit remain
deferred.

## Pinned Near-Start Opponent, Two-Arm Test

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

**Declared and run on 21 September 2026.** The tuning study's
confirmations show that 46 of 48 selected models won no confirmation game from
their weaker spawn end, one more won a single game there, and the habit is
formed by the first checkpoint. Mirrored self-play cannot penalize a
fixed world direction, because the opponent is a copy of the learner and always
meets it. This test asks whether keeping a near-untrained copy of the learner as
a permanent training opponent changes that.

Both arms use the c03 settings: 512 environments, rollout length 32, entropy
0.01, gamma 0.999, actor/critic learning rate 0.00025, dense score-delta
shaping 0.01, input scale 0.01, canonical 5v5, all training maps, K20/H300, and
exactly 19,999,744 real transitions with 1,221 updates. Seeds are 19,044,701,
19,044,702 and 19,044,703, the study's seeds, so each arm shares its starting
weights with the study. The treatment adds one setting, `pinned_opponent_share`
0.1: the actor after the first completed update is captured into history slot 0
and stays there; each new game meets it with probability 0.1, another stored
snapshot with total probability 0.2 once any exist, and current weights
otherwise. The 5% to 95% captures keep their timing; the never-played 100%
capture is dropped. The runner derives the matching schedule switch,
`early_history_capture`, which is also a public `make_training_schedule`
argument; the collection refuses a mismatched pair. The control keeps the
unchanged 80/20 recipe.

The control arm reuses the study's three finished c03 runs only if an
equivalence job passes: the control config, unchanged, trained on the new
source with the default share to its first checkpoint at 2,015,232 steps must
reproduce the study run's weight digest, inference digest, first 123 update
rows and 40-game routine cells exactly. If it does not, a second run from the
study's own frozen package separates a code fault from GPU nondeterminism, and
the control arm is trained again in the new package. That decision is fixed
before launch and recorded in the package declaration.

Routine checks, checkpoint selection and the 200-game confirmation are the
study's rules, unchanged. The primary readout is declared before launch: for
each treatment seed, the weaker spawn end's confirmation wins out of 100.
Control weak ends won 0 of 100 in every seed. Success needs at least 10 in all
three seeds; partial needs one seed at 10 or all three at 3; otherwise null.
Guardrails report total wins per seed (at least 30, or the result is an
approved tradeoff), losses (at most 5) and the realized pinned share (0.08 to
0.12). Low-kill counts describe games where each side scored at most one kill;
they are not a measured absence of contact.

Actual cost with the reused control: the three treatment runs and the report
took 53.0 minutes. The equivalence job took 144.7 s and the engineering check
159.6 s on the sealed preparation (142.4 s and 151.2 s on the first, superseded
one), and the CPU replay check 31 s. Three seeds per arm cannot establish the
effect; results are reported per seed with no significance claim. c03 was chosen
from sixteen settings on the same confirmation numbers, so a treatment tie is
confounded; a treatment win is the stronger result. A failed realized-share
guardrail voids the verdict; the package declared that before launch. The
package, declaration, configs, hashes, commands and reports live under
`artifacts/m9-m10/packet-4/pinned-opponent-20m/`; the replay check lives under
`artifacts/m9-m10/packet-4/c07-s701-spawn-replays/`. The full repository gate
and commit remain deferred.

**Result (2026-09-21).** Verdict by the frozen rule: NULL. Every
treatment seed won 0 of 100 confirmation games from its weaker spawn end, as did
every control seed. All guardrails passed and the realized pinned start share
was 0.098 in each seed, so the declared treatment was delivered. Total
confirmation wins were 60, 63 and 74 against the control's 57, 45 and 47, and
the selected checkpoints came earlier (6.0, 12.0 and 14.0 million steps against
12.0, 20.0 and 20.0 million). c03 was chosen on those same numbers, so this says
the recipe did not hurt learning, not that it helps. One secondary signal: in
two of three treatment seeds the weaker end's mean kills per routine game rose
over the last three checkpoints (seed 701: 2.6, 2.1, 6.9; seed 703: 3.1, 5.7,
6.8; twenty games each) while no control seed rose above 1.6, and weak-end
low-kill games fell to 4 and 1 of 20. That is a direction seen in twenty-game
checks, not a result. The pinned share stays available at its zero default and
the shared recipe is unchanged. Tables, per-checkpoint spawn-end splits and
plots: `artifacts/m9-m10/packet-4/pinned-opponent-20m/reports/results.md`.

| Arm | Seed | Selected Step | Wins /200 | Strong End /100 | Weak End /100 | Weak-End Low-Kill /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 19,044,701 | 19,999,744 | 57 | 57 | 0 | 82 |
| Control | 19,044,702 | 19,999,744 | 45 | 45 | 0 | 69 |
| Control | 19,044,703 | 12,009,472 | 47 | 47 | 0 | 21 |
| Treatment | 19,044,701 | 12,009,472 | 60 | 60 | 0 | 82 |
| Treatment | 19,044,702 | 14,008,320 | 63 | 63 | 0 | 65 |
| Treatment | 19,044,703 | 6,012,928 | 74 | 74 | 0 | 66 |

## Mirror Check: Post-Hoc Canonicalization Of One-Sided Models

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps: maps 0 to 41 with revision 6 of Map 39, before its
update. They are historical only. Do not use them as current evidence, for
comparisons or to choose settings.

**Declared and run on 21 September 2026.**
Every one of the 52 maps is an exact left-right mirror image of itself (largest
deviation 7e-7 world units, float rounding), the two spawn banks sit at x=0.5
and x=19.5, and the actor observation is in raw world coordinates with no team
label. A policy that learned one spawn end can therefore be shown the mirrored
world whenever its team starts on the other end, with the move it chooses
mirrored back before the game receives it. The wrapper flips the x of every
active unit row, both spawn-pad banks, the previous-move one-hots and the move
mask (East and West, Northeast and Northwest, Southeast and Southwest change
places), leaves the obstacle table as it is because every map equals its own
mirror image, and touches nothing else: targets are roster slots, Ultimates a
bit, every ability target-relative, and there is no facing or velocity. The flag
is recomputed each decision from the actor's own spawn pads and the map width,
so memory keeps its shape, Team A and Team B are treated alike and no
information right widens. This is baseline-side code around the researcher's
System; the simulator, environment and evaluator are unchanged.

Three one-sided models are played plain and mirrored: c07 seed 19,044,701 and
t03 seed 19,044,703 (both learned the right end) and c03 seed 19,044,702 (left),
each in 200 paired Random games on maps 42 to 46 at root 19,044,791, K20/H300.
Frozen rule: per model, the weak end is the end with fewer plain wins and S its
plain strong-end wins; SUCCESS when every model's mirrored weak end reaches half
of S while its mirrored strong end keeps 70 percent of S; PARTIAL when two
models reach 10 mirrored weak-end wins; NULL otherwise. Because the maps are
exact mirrors, a correct transform must carry the strong-end rate to the weak
end, so the run proves the transform and reports the real per-end numbers at
once; a large shortfall means a transform fault or a hidden asymmetry. A SUCCESS
says nothing about what a model trained with the mirror on from the start
learns, or about stronger opponents. Package, declaration, tests and results:
`artifacts/m9-m10/packet-4/mirror-check/`.

**Result (2026-09-21).** Verdict by the frozen rule: SUCCESS. Wrapped,
every model won from its formerly dead end at about its learned-end rate, and
its learned-end games were unchanged because the wrapper does nothing there.
Mirrored weak-end kills per game were 18.3, 18.9 and 16.3 against 0.8, 1.1 and
1.2 plain, and the plain weak ends' 81, 66 and 69 low-kill games fell to zero.
No model lost a game in any pass. The one-sided habit is therefore a frame
effect: the policies learned one side of a symmetric world and never learned the
other, and a mirror of the inputs removes the symptom without any training.
Timing: a mirrored pass took 19 s against 7 s for a warm plain pass, but the
mirrored program compiled fresh in that pass, so the per-step cost of the flip
is not separated here and is measured in the follow-on packet. Limits: three
models, 100 games per end, Random opponent, training maps; a model trained with
the mirror on from the start is a separate question, as are stronger opponents.

| Model | Learned End | Plain End 0 | Plain End 1 | Mirrored End 0 | Mirrored End 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| c07 seed 19,044,701 | right | 0 | 66 | 62 | 66 |
| t03 seed 19,044,703 | right | 0 | 74 | 73 | 74 |
| c03 seed 19,044,702 | left | 45 | 0 | 45 | 41 |

## Spawn Frame Option For Recurrent MAPPO

**Superseded, 22 September 2026.** The results in this section come from models
trained on the old maps (maps 0 to 41 with revision 6 of Map 39) and from games
against the old scripted controllers ALPHA version 2 and BETA version 4. They
are historical only. Do not use them as current evidence, for comparisons or to
choose settings.

**Declared on 21 September 2026; approved by the owner. First package run the same day as a pilot; second package declared below.**
(Status, 22 September 2026: `"left"` is now the default and the value
`"right"` has been removed; the text below records the setting as declared.)
The mirror check above showed the one-sided spawn habit is a frame effect. The
recurrent MAPPO baseline now carries one setting, `PPOConfig.spawn_frame`, with
values `"world"` (today's raw coordinates, bit for bit), `"left"` (the actor's
permitted view is reflected whenever its own team starts on the right bank, so
every game looks like a left start, and the chosen move is reflected back) and
`"right"`. The transform lives in the baseline's input adapter and in three
small public helpers and the `MOVE_MIRROR` table beside the team-view builder;
the environment, evaluator and
tournament never apply it, and a researcher's own System may use the helpers or
not. The precedent and the design claim are recorded in the source-reuse ledger:
Google Research Football flips the live observation and translates the action
back for right-side players, RLGym inverts physics for the orange team to avoid
re-learning the same strategy on both sides, and AlphaZero orients the board to
the current player. The setting is saved in the actor export and its inference
identity; old exports mean `"world"`; a loaded actor plays in the frame it was
trained in.

The experiment: the c03 settings with `ppo.spawn_frame` `"left"` and a fresh
random initialization result, seeds 19,044,701, 19,044,702 and 19,044,703,
19,999,744 real transitions, the study's checkpoints, 40-game routine checks and
200-game confirmations at root 19,044,791, split by spawn end. Control: the
study's finished c03 runs, reused under an equivalence job that must reproduce
seed 19,044,701 to its first checkpoint under `"world"`; that reuse is bounded
by one recorded limit, the mirror check's plain replays of these same
confirmations matched 599 of 600 recorded games, so GPU replay is exact up to
contact and not beyond. Frozen rule: TWO-SIDED
when every seed's weaker confirmation end wins at least 10 of 100 and at least
half of its stronger end; WITHIN THE DECLARED PERFORMANCE ALLOWANCE when every
seed's task score over 200 games (win 1, draw 0.5, loss 0) is at least the
control seed's minus 0.05, losses are at most 5 and wins are not zero; the
default becomes `"left"` for recurrent MAPPO only when both hold and every case
completed; PARTIAL keeps `"world"` and the option when TWO-SIDED holds in
exactly two seeds with the allowance met everywhere, or TWO-SIDED holds
everywhere and the allowance fails in exactly one seed; NULL otherwise, which
for an exact
transform points first at the transform. Three seeds, reported per seed, no
significance claim; c03 was chosen on the same confirmation numbers, so a tie is
confounded. A policy trained in a canonical frame is two-sided by construction;
the run measures whether it is as good as, or better than, the world-frame
control, not whether it is symmetric.

### First Package: Pilot Result, Audit And Correction

**Run and read on 21 September 2026.** The three treatment seeds
trained and confirmed. Against Random (200 games, 100 per end): seed 701
selected its 10.0M checkpoint and won 1 game with no losses, about 6.3 kills
and 4.6 deaths per game on both ends, task score 0.5025 against the control's
0.6425; seed 702 selected 4.0M and won 79 (43 and 36 by end) with no losses,
16.5 kills and 2.7 deaths per game on both ends, task score 0.6975 against
0.6125; seed 703 selected 12.0M and won 44 (27 and 17), 14 kills and 1.5
deaths, task score 0.6100 against 0.6175. By the frozen rule the verdict is
NULL, carried by seed 701 alone: seeds 702 and 703 are two-sided and within
the allowance. Seed 701's routine checks show why: it played both ends alike
at every checkpoint but never rose above a brawl (up to 8.9 kills against 5.6
deaths per game), where the world-frame control's strong end climbed to 18.5
kills against 1.6 deaths while its weak end stayed passive at about 1 kill and
0 deaths. The working explanation is self-play: a world-frame policy fights a
passive copy of itself in half its games and sharpens a safe attack that also
beats Random; a frame-trained policy meets a two-sided copy of itself in every
game. This is one seed's trajectory, not a property of the frame.

**Head-to-head, same day.** Each seed's treatment (final 20M actor)
against the study's control final actor wrapped two-sided with its learned
bank, 100 paired games in each team order on maps 42 to 46 at root
19,045,101: seed 701 even (4 wins 3 losses and 3 wins 2 losses, 93 and 95
draws) despite 1 against 57 wins on the Random proxy; seed 702 even to
slightly treatment (20 wins 15 losses and 16 wins 16 losses); seed 703 the
control (0 wins 22 losses and 28 wins 0 losses). Against the scripted teams
every one of the six models lost 95 to 100 of 100 to tdm-alpha and to
tdm-beta, which reached 20 kills in nearly every game; our models scored 2 to
11 kills per game against them. Reading: the Random proxy ranks farming a
passive opponent, not fighting; the scripted teams are the strongest opponents
available today by a wide margin; selection and validation should move to
strong opponents, which is a separate packet.

**Audit and correction, same day.** Four independent auditors ran
fresh code on the trained weights. The learner's update recomputes every
stored log probability on real fights to float rounding; both teams' self-play
actions go through the frame from their own pads; masks and legality hold on
every row. One material defect: reflecting obstacle rows in place made the
reflected table a row permutation of the authored one on every map with
obstacles, and the encoder reads rows in slot order, so the two ends did not
see the same input vector (logit differences of 0.02 to 0.17 at the first
step, a few percent by first contact). The helper now leaves any obstacle row
whose mirror image is already in its table as authored and reflects only rows
without a partner; on all 52 maps the table is unchanged and the encoded
vectors of the two ends are identical (difference 0.0), and asymmetric layouts
keep correct geometry. The first package therefore stands as a pilot under an
impure transform. Its evidence lives in
`artifacts/m9-m10/packet-4/spawn-frame-20m/` (qualification record,
head-to-head tables, audit disposition in the private brief).

### Second Package, Declared

**Superseded, 22 September 2026.** This package has since run on the old maps
(maps 0 to 41 with revision 6 of Map 39). Its result and its games against ALPHA
version 2 and BETA version 4 are recorded in its package and are historical
only, like the rest of this section.

**Declared on 22 September 2026; not yet run.** The same three-seed
two-arm test on the corrected transform:
`artifacts/m9-m10/packet-4/spawn-frame-v2-20m/`, same settings, seeds, budget,
control reuse under the equivalence job, and the same frozen rule. Its post-hoc
check must reproduce the mirror check's numbers exactly (62/66, 73/74, 45/41),
because the shipped helper now makes the choice the mirror check's wrapper
made. Alpha and beta games and the head-to-head follow the run as descriptive
evidence.


## Current Recurrent MAPPO Search: Declared 23 September 2026

The [search protocol](mappo_search.md) defines a new, replicated comparison
under the current maps, left spawn frame and fixed Alpha validation. It uses
value normalization for every recipe. Earlier world-frame, Random-validation
and older-controller runs retain their original identities and are historical
evidence. They are not reused as fresh seeds in this study.

The protocol searches update reuse, clipping, batch/window length, actor
learning rate, entropy and Alpha pin share. These are important controls in
the [original MAPPO study](https://arxiv.org/html/2103.01955v4). The recipe
comparison is not a factorial estimate of every interaction. Critic learning
rate, architecture, rewards and normalization constants remain fixed.

Value normalization follows the official implementation, with the source and
explicit JAX adaptations in the [reuse record](source_reuse.md). An independent
reference checks moving statistics and normalized losses; the existing Mava
reference remains in place with normalization disabled. The same critic pass
supplies raw values for GAE and exact normalized predictions for value clipping.
Actor exports retain only their original actor information.

A bounded internal RTX 5090 check at B512/T32, four epochs and actual Alpha
share 0.1 measured 18,797 real transitions per second with normalization and
19,001 without. Both arms used gamma 0.99; the declared study uses 0.999 and
requires its own full timing calibration. The observed difference is about 1.1%,
but the learned trajectories also diverge, so this is not a pure isolated
normalizer cost. Median learner updates were 0.219 and 0.218 seconds. Both arms
had the same collection-program hash and sampled process GPU memory near
5.50 GB. Each arm used five warm samples after actual episode resets.

A separate B32 composition check matched the numerical reference on two
successive changes of weights and normalization statistics. Integer fields
and keys matched exactly; floating values used the existing declared tolerance.
Collection and update each reused one compiled program in those checks.
The added clipping-anchor array uses 327,680 bytes at B512/T32 and the running
statistics use 12 bytes; these counts are distinct from measured peak memory.

These are engineering checks, not evidence of better sample efficiency or
learned tactics. Raw qualification records, source identities, the timing-only
tier choice, all attempted runs and generated results belong under
`artifacts/m9-m10/packet-4/mappo-optimization-20260923/`. No study outcome is
claimed here before the detached study completes. The driver must refuse to
start if its smallest approved replicated comparison cannot fit the budget.


A separate controller audit kept identical permitted B512 inputs and swapped
only the three controller files. The approved blocked-wall correction cost
about 1.2–1.6 milliseconds per full all-actor call, an observed 6.6–8.7% increase.
The played-state inputs exercised changed decisions. Both copies reused their
compiled call and had sampled process GPU peaks near 1.74 GB. This resolves the
earlier failed range criterion as a measured correctness/performance tradeoff;
it does not establish equal speed or a whole-training slowdown of that size.
The raw comparison and limits are in `qualification/controller-cost-report.md`
under the study artifact root above.

Pinned Policy adapters now use the smallest fitting quarter, half or full
batch, rounded up. Zero pinned games skip their call. Generic Systems still
receive the full batch with other games marked invalid. The wrapper reuses the
same input, key, memory and action handling for both compact sizes; there is no
new public setting.

A matched B512/T32 check motivated the middle size. Ordinary c05 training at
pin share 0.3 reached 161 Alpha games. From two saved states and identical
random keys, a temporary half-size reference reduced whole-collection time from
0.60044 to 0.54963 seconds during the reset wave, and from 0.66718 to 0.58389
seconds immediately afterward. All 332 output arrays and scalar values matched
exactly in each case. Both collectors reused one compiled program and had 5.488
GB sampled peak GPU memory. This measures an avoidable cost at those reached
states, not a full-training gain. The actual quarter/half/full route was then
checked separately. The raw reference, failed first harness attempt, correction
and limits are in `qualification/pin03-capacity-report.md` under that same
study artifact root.

The actual three-size route also matched every output in those two 0.3 cases.
Collection time fell by 8.25% and 11.67%. A separate natural 0.1 run reached 49
pins; old and new collection times differed by +0.18% and -0.03% in its two
matched cases, within the variation of three timing samples. No material warmed
regression was observed in that check. All four cases matched 332 output arrays
and scalar values exactly and reused one collector program per process.

The extra branch took about 5.7 more seconds to compile and 0.5 more seconds to
prepare compiler input. It added 41,728 temporary GPU bytes and 0.27–0.28 GB of
peak host RAM. Sampled peak GPU memory stayed at 5.488 GB. This cold cost
applies to every pinned-Policy recipe, including those that keep using the
quarter route. These collection measurements do not establish a net whole-study
speedup; final calibration must price the actual source. Raw results, source
identities, the explicit paired engineering state transfer and limits are in
`qualification/adaptive-capacity-report.md` under the study artifact root.
