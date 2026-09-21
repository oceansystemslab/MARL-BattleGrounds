# Development Checkpoint Population

**Start here for the saved models and manuscript evidence.** On 20 September
2026, we adopted every model in the completed checkpoint comparison as
**Development Population V1**. Keep this page as the entry point as the population
grows. The [baseline methods record](baseline_methods.md) owns the tuning history.

## Current Population

| Source | Members | Training Seed | Environments / Rollout | Reward | Input Scale |
| --- | ---: | ---: | --- | --- | ---: |
| Recent curriculum + reward shaping run | 10 | 19,044,601 | 512 / 32 | Dense kill/death reward, coefficient 0.01 | 0.01 |
| Recent reward shaping only run | 10 | 19,044,601 | 512 / 32 | Dense kill/death reward, coefficient 0.01 | 0.01 |
| Earlier ten-hour curriculum run | 20 | 19,042,001 | 1,024 / 128 | Potential shaping, coefficient 0.01 | 1.0 |
| Shared untrained neural actor | 1 | 19,044,601 | No training | No learned experience | 0.01 |
| **Total** | **41** | | | | |

These are **40 trained snapshots from three runs**, plus one untrained reference.
The three trained runs use two distinct seeds. Snapshots from one run are related
models, not independent training seeds. All use the 128-wide recurrent MAPPO
actor shared across teammates, with separate actor memory. The untrained actor
is not the built-in Random policy. Its original export comes from the B512/T64
screen; its verified initial inference identity is shared with the later pair.

Membership keeps early, weak and final checkpoints. It was not filtered by wins
or test-map outcomes. Earlier B/T trials and mini checks remain in the wider
archive; V1 names the 41 actors in the latest comparison table.

## Find A Model Or Result

| What You Need | Where To Open |
| --- | --- |
| Exact membership and identities | [V1 JSON record](../../artifacts/m9-m10/packet-4/model-catalog/development_population_v1.json) |
| One row per member: settings, scores, times and model paths | [V1 CSV](../../artifacts/m9-m10/packet-4/model-catalog/development_population_v1.csv) |
| Readable ranked checkpoint table | [All checkpoint results](../../artifacts/m9-m10/packet-4/curriculum-vs-rs-30min/reports/all_checkpoints.md) |
| Learning curves for all three runs | [PNG](../../artifacts/m9-m10/packet-4/curriculum-vs-rs-30min/reports/checkpoint_comparison.png) · [SVG](../../artifacts/m9-m10/packet-4/curriculum-vs-rs-30min/reports/checkpoint_comparison.svg) |
| Why we chose B512/T32 | [Decision](baseline_methods.md#current-mappo-standard) · [All 17 B/T results](../../artifacts/m9-m10/packet-4/bt-extra-5min/reports/combined_results.md) |
| Curriculum versus reward shaping only | [Methods and results](baseline_methods.md#curriculum-and-dense-reward-comparison) |
| All twenty versions of the old ten-hour model | [History results](../../artifacts/m9-m10/packet-4/c-rs-10h-history-validation/README.md) |
| Earlier diagnosis, input scaling and reward trials | [Diagnosis](../../artifacts/m9-m10/packet-4/learning-diagnosis/README.md) · [Task reachability](../../artifacts/m9-m10/packet-4/learning-diagnosis/task_reachability_bounds.md) |
| Why we run one learner at a time | [GPU sharing measurements](../../artifacts/m9-m10/packet-4/b512-t32-concurrency/README.md) |
| Earlier models outside V1 | [Dated model inventory](../../artifacts/m9-m10/packet-4/model-catalog/README.md) |

The JSON record binds each model's weights, input scale, export, originating
checkpoint, source run, seed and step. It includes run settings, dependencies,
source/content hashes, and links and hashes for validation summaries and raw M8
game files. The CSV keeps the manuscript columns together. Catalog links start at
the repository root; copied original metadata retains its original paths. This
is an evidence catalog, not a trainer config API. Preserve weight identity and
inference identity as separate fields:
the latter also binds input scaling.

The V1 JSON SHA256 is
`0bb39b813048f681e481f34268c070ec0de935c68814522bfac9971d018d7a38`.
Name this exact population version in future experiment records.

Models remain at their recorded locations. This index did not copy or move them.
These are saved actors for play; not every member has a retained learner state
from which training can resume.
The `artifacts/` files are local and ignored by Git; the catalog is not an
off-machine backup or a published release. Preserve actors, source packages and
raw records together when preparing the manuscript archive.

## Reading The Evidence

Each member has the same 40-game development check against Random: maps 42–46,
root seed 19,043,001, four seed pairs per map, canonical 5v5 and K20/H300.
There are 1,640 games across 41 unique actor checks. References to the shared
initial result or old-final result do not count as newly played games.

**Both spawn arrangements are covered.** At B512, training uses 256 environment
lanes with normal spawn ends and 256 with exchanged spawn banks. Each lane keeps
its arrangement through resets; the learner stays Team A. Validation plays each
map/seed pair from both ends: 20 games per arrangement. No game changes ends
halfway through.

Observed peak win rates were 32.5% for reward shaping only, 22.5% for the recent
curriculum run and 12.5% for the old run. These peaks were selected using the same
forty games; they have no fresh confirmation yet. Performance against Random does
not establish how these models rank against one another.

Native task score is `(wins + 0.5 * draws) / games`. The exploratory combined
score is `0.5 * mean kill difference + 0.5 * wins`, where wins is the count out of
40. It is not a normalized 50/50 benchmark score. Keep W/D/L, kills and deaths
visible too. Tables rank by win rate, then fewer losses and earlier step. Game
uncertainty preserves paired games; it does not measure training-seed variation.

Warm training time and speed exclude the first compiled block, checks and saving.
Total run time includes recorded overhead. Checkpoint capture times precede their
own validation; final totals include final outputs. Old-history evaluation was
performed after training. The recent RS-only run briefly shared the GPU with the
earlier old-final check, so its recorded time is not fully isolated.

The old final actor was inspected on test maps 47–51 against Alpha, Beta and the
20-million-step actor. Keep this exposure in the
[replay record](../../artifacts/m9-m10/packet-4/c-rs-10h-test-replays/README.md).
Do not use those test results to choose membership or sampling weights. No
individual test evaluation of the other nineteen history actors is recorded in
the reviewed evidence. This development archive does not replace the official
validation panel, Big N selection or locked-test protocol.

## Growing And Using The Population

Keep V1 unchanged. Add models through a new version naming its parent, exact
membership and reasons for changes. Keep member IDs and original evidence.
Duplicate exports of the same inference identity should not gain sampling weight.
Each experiment pins one version and declares sampling weights before starting.
Later additions do not change running experiments. Scores against changed
opponent sets belong to separately labelled evaluation series.

**Proposed next experiment, not enabled yet:** at episode reset, choose current
self-play with probability 80% and a frozen population member with probability
20%. This does not guarantee 20% of transitions against the population. Declare
the weights inside that population before launch. Uniform member sampling gives
the old run 20 of 41 choices; equal weight per source run is a different recipe.
Whether to sample the untrained reference also remains an experiment choice.

The current trainer uses 80% current self-play and 20% its own saved history,
with twenty history slots. It does not load this external population. Integration
must preserve each member's input scale and inference settings; mixing weight
arrays alone would change the older policies. Choose frozen opponents at reset,
retain their weights for the game and keep recurrent memory separate.

Once models become training opponents, games against them measure performance
against familiar opponents. Keep a separately declared evaluation set for claims
about unfamiliar opponents. Existing checks remain development evidence; they
do not establish that the proposed 80/20 change improves MAPPO.

## Manuscript Use

Start with this page and the [methods record](baseline_methods.md). Cite the
population version, original run identities and source hashes. Use the CSV for
tables and the per-member validation links for raw games. Report the search,
failed trials, later selection decisions, seed reuse, actual timing, test exposure
and changes between old and new training recipes.

All 41 actor payloads and their evaluation identity joins were checked when V1
was created. This was a file check; it ran no models or new games. The full
software gate, final baseline replication, unseen-opponent claims and release
packaging remain separate work.
