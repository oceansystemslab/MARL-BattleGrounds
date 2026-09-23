# Recurrent MAPPO Hyperparameter Search

This eight-recipe study gives recurrent MAPPO a finite search under the current
spawn frame, maps, Alpha opponent and value normalization. It measures complete
recipes. It does not claim to find a global optimum.

The ordinary trainer owns learning and checkpoints. The ordinary evaluator
owns games, identities and recovery. The study driver adds recipe ordering,
timing-based budgets, three-seed comparisons and a detached supervisor. It does
not introduce another model format or change the `System` contract.

## Fixed Choices And Search Recipes

Every run uses shared recurrent actors, a separate centralized critic, width
128, left spawn coordinates, value normalization, 512 environments, canonical
5v5, K20/H300 and uniformly eligible training maps 0–41. Actor information
limits stay unchanged. Training uses native reward plus `score_delta` shaping
at 0.01. Curriculum and recording are off. Gamma is 0.999, GAE lambda is 0.95,
input scale is 0.01, and the existing current/history opponent rules remain.

The reference uses actor and critic learning rates 0.00025, four epochs, two
minibatches, two gradient groups, clipping 0.2, entropy 0.01, 32-step rollouts
and Alpha for 10% of new training games. That percentage describes starts;
saved exposure records separately show the fraction of actual transitions.

| Recipe | Change From The Reference |
| --- | --- |
| c00 | Reference |
| c01 | Actor learning rate 0.0001 |
| c02 | Entropy coefficient 0.003 |
| c03 | One minibatch |
| c04 | Clipping 0.1 |
| c05 | Alpha for 30% of new games |
| c06 | 64-step rollouts |
| c07 | Two epochs |

Clipping changes the existing shared policy/value clipping setting. Longer
rollouts change the recurrent gradient window, the batch size and update rate
together. These comparisons must not be described as isolated causal effects.

## Budgets And Selection

Before scientific training, bounded engineering probes measure the actual
qualified code. Their learners and games are excluded from the study results.
The current declaration fixes recipes c00–c07 and three discovery training
seeds per recipe. Earlier packages retain their original 12/10/8 tier rules;
this amendment does not change their source, declarations or evidence.

All 24 discovery runs receive exactly 20,054,016 real environment transitions. The budget is a multiple of 131,072, so quarter
checkpoints occur at the same experience counts for both rollout lengths.
The reference and best complete challenger then receive three fresh seeds each
at twice that budget. These runs start fresh. They do not continue a discovery
run with a changed opponent-history schedule.

The six fresh finalist runs each receive exactly 40,108,032 transitions. This
is 30 training runs and 721,944,576 transitions in total. The forecast includes
setup, training, saving, validation, assessment and reporting, with a 10% timing
margin and five-minute reporting reserve. It must fit thirteen hours. Spare
time does not increase experience, and slow timing does not reduce it. The
fixed_discovery_steps declaration field makes this explicit; declarations
without it retain the historical adaptive rule. Rewards cannot change timing
admission or experience budgets.

Each training-run forecast adds its measured setup, training, checkpoint and
report costs, one first-validation setup cost, five warm routine passes and
at most three warm confirmation passes. Engineering checks repeat each
validation workload with changed actor weights and require reuse of the
evaluator's compiled program. The first and repeated timings are both retained.
Assessment costs come from separate fresh workers, including their startup and
cleanup, because each scientific assessment also starts in a fresh worker.
Training setup includes the supervisor's process start through interpreter,
imports and JAX initialization, followed by the separately measured trainer
setup. These two intervals do not overlap.

Discovery checks initialization and 25%, 50%, 75% and 100% of training against
fixed Alpha on validation maps 42–46, with both spawn ends: 40 games each.
The shared selection code shortlists the best two trained checkpoints and the
final checkpoint when distinct, then plays 100 fresh confirmation games for
each. Initialization cannot win. Fresh finalist runs use 80 routine games and
200 confirmation games per shortlisted actor.

Recipe selection uses mean confirmed native task score across all three seeds,
then mean kill difference, earlier mean selected step and recipe ID. Native
score is win 1, draw 0.5 and loss 0. Maps receive equal weight. A missing or
failed training seed makes the recipe ineligible; it is never averaged away.

The final recipe choice and all six finalist actor identities are frozen before
400 new Alpha games and 200 new Beta games per actor. These results cannot
change selection. Beta is a related familiar diagnostic, not an untouched
opponent. Protected test maps and scenarios are excluded.

## Prepare, Qualify And Launch

Use the training, visualization and CUDA extras and the internal RTX 5090.
The command below copies current public working files, including reviewed
uncommitted changes, into an immutable package and installs the locked runtime.
Private milestone files and Git state are not copied or changed.

```bash
uv run python -m marl_battlegrounds.training.search prepare \
  artifacts/m9-m10/packet-4/mappo-optimization-20260923/package-eight \
  --repository "$PWD" \
  --gpu-uuid GPU-6b11a0c4-14e8-6782-8932-1df56d599796

bash artifacts/m9-m10/packet-4/mappo-optimization-20260923/package-eight/calibrate.sh
bash artifacts/m9-m10/packet-4/mappo-optimization-20260923/package-eight/launch.sh
```

Preparation and calibration do not start the scientific study. Inspect the
qualified source, measurements and resolved `budgets.json` before launch.
Calibration runs in the foreground and has its own one-hour limit. The thirteen-
hour study clock begins only at scientific launch. Launch returns immediately;
the detached supervisor continues without an assistant or open terminal.
An interrupted calibration needs a new package. Resuming a half-finished
timing probe would omit some cold setup cost. Clean completed probes can be
reused when the same calibration command is explicitly repeated.

Use the package's `status.sh`, `stop.sh`, `resume.sh` and `report.sh` commands.
Verbose training is on by default. Each job writes human-readable progress to
`jobs/<job-name>/worker.log` about every ten seconds and at phase changes.
`status.sh` prints saved JSON progress; launching does not keep a terminal open.
Status reads saved files and process identities without starting JAX. Stop asks
the verified owned processes to shut down. Resume keeps the original source,
recipes, seeds, experience budgets and clock. It recovers through the trainer
and evaluator rather than repeating complete work or replacing failed seeds.

Numerical work stops by twelve hours fifty-five minutes; the remaining five
minutes are for cleanup and reporting. Thirteen hours is the admission and outer limit;
experience budgets remain fixed. Shutdown starts early enough to include its
bounded waits. The outer supervisor gives each inner supervisor time to stop
its own worker group before forcing the controller group to exit. A stopped
run retains its last durable
checkpoint; forced shutdown does not promise a new checkpoint. An expired
resume may produce reports but cannot train again. Infrastructure, identity,
storage or evaluator failures stop the study and preserve their error.

One worker runs at a time. Command-line controllers and supervisors use CPU
for bookkeeping, including actor identity checks. Only numerical workers select
the GPU. Direct Python calls place newly imported metadata constants on CPU and
restore the caller's device default; JAX may still discover other visible
backends in that caller. Status stays lazy and does not import JAX.

GPU selection uses its recorded UUID, not a changing device ordinal. The
package forces `XLA_PYTHON_CLIENT_PREALLOCATE=false`: it does not reserve 75%
of GPU memory up front. It keeps the existing 0.85 allocator setting and its
own compilation cache. Memory grows as needed, and JAX retains buffers for
reuse; this is not a promise to release every unused byte between updates. Qualification must establish actual memory and
disk headroom; these settings are not measurements of memory use.
The disk forecast counts five candidate checkpoints, two recent recovery
checkpoints, exported actors, growing logs and metadata, and every planned
validation and assessment output. It uses measured probe bytes, then adds 25%
and 2 GB for variation, temporary files and reports. This is a reservation based
on measured outputs, not a proof that every future file has a fixed size.

## Evidence And Interpretation

The package saves `declaration.json`, `calibration.json`, `budgets.json`, source
hashes, dependency/runtime identity, frozen opponent panels, the original clock,
every attempt and all per-run records. Selection records bind their source rows
and actor identities. Raw games retain maps, spawn ends and their actual keys.

The report contains every recipe and seed, including failure, selected and
final actors, one set of shared learning curves per recipe and phase, validation
map cells, assessment cells and cost/exposure records.
The frozen final selection binds all six selected exports and their final-run
exports to verified file hashes, originating checkpoints, runs and seeds.
Assessment checks the saved selected identity before creating or recovering
its game writer, then checks the returned summary before saving it. Replacing
an export with another valid actor fails these checks; its path alone is never
treated as proof of identity. Resumed results must match the same frozen request.

`recipe_summary.csv`
reports means and sample standard deviations only for eligible three-seed
comparisons. Report environment transitions and living actor decisions
as separate counts. Explain setup and compilation separately from warmed
training, and include validation, output, RAM and VRAM costs in the complete
workflow. Losses in normalized value units must be labelled accordingly.

Paired-game uncertainty keeps both spawn ends together. It describes the fixed
saved actors. The three training seeds supply a separate, small sample of
training variation. Fresh assessment games do not make the training seeds used
to select a recipe an untouched final benchmark. Plot all seed results rather
than only the best curve. A fast simulation, low value loss or favorable Alpha
score alone does not prove sample efficiency, tactics or cooperation.

Other baselines should follow the same principles: choose a finite set of
method-appropriate settings, freeze fair experience budgets and selection rules,
use independent training seeds, retain failed trials, and assess frozen choices
on fresh games. Their algorithm-specific machinery and settings remain their
own; they do not inherit MAPPO's optimizer or rollout rules.
