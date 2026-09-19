# MARL-BattleGrounds Evaluation Protocol

## Status and authority

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** This is the controlling public
> evaluation protocol.

This document sets the rules for evaluating MARL-BattleGrounds policies. It
says which policies and conditions a result covers, how to combine results,
how to report uncertainty, and how to select checkpoints. It also owns actor
information records, cross-play, controlled scenarios, runtime measurements,
and the treatment of failures and unobserved endpoints.

**Current reading rule, 2026-09-17:** a protocol requirement is not proof of
scientific qualification. The package implements native reset/step and Systems,
evaluation, custom and canonical tournaments, verified game reuse, headlines,
recording, recorded restart, result reading, commands and portable replay viewing.
Separate maintainer admission and release machinery is tested with fixtures.
There is no qualified official Big 12 bundle or trained baseline population.
Official numerical game/resource budgets and the remaining approval gates stay open. This update
supersedes the older weekly review and fixed 100-game official budget; it does
not change old records or the current custom runner's default of 100.
Current milestone numbers follow
[A36's executive roadmap mapping](../design/specification_amendments.md#a36-submission-roadmap-approved-tdm-content-and-m7-closeout).
Historical amendment titles and anchors retain their original milestone numbers.
Gymnasium/PettingZoo adapters are deferred until a specific learner integration
shows a need. Optional dependencies do not implement those adapters. M9 owns
training distributions and curricula; M10 owns learners and learning experiments.
The growing-pool example checks API integration, not an approved M9 curriculum.
The companion [metric specification](metric_specification.md) owns stable
metric meanings, sufficient components, eligibility, attribution, and allowed
interpretations. Accepted departures from the original design PDF are recorded
in the [specification amendments](../design/specification_amendments.md).

Researchers choose their experiments. Each evaluation suite and experiment
manifest must record its choices before the locked evaluation starts. A reader
must be able to reproduce the result and see exactly what it supports.

## Protocol objects

Four kinds of record separate a measurement's meaning from the experiment
that uses it. These are contract roles; they do not imply that every named
record is a public Python class or that a universal registry exists.

### Metric definition

A versioned metric definition keeps the same meaning across studies. It names
the question, who or what is measured, which recorded facts it uses, and its
units. It states when an opportunity counts, which raw values must be kept,
how to combine them, and what a zero denominator means. It also limits credit
and causal claims, names checks against misleading results, and records whether
the metric is ready to use.

Changing a formula, denominator, eligibility rule, amount stage, attribution,
or semantic scope creates a new metric version. Changing an opponent pool,
cell weight, confidence interval, or checkpoint does not.

### Evaluation suite

An evaluation-suite declaration, conceptually `EvaluationSuiteV1`, fixes:

- suite ID, version, canonical digest, ownership classification as `official`
  or `custom`, separate canonical-condition status, and intended scientific
  claim;
- battleground identity and version plus the resolved episode-configuration
  contract;
- execution-information provenance; an official suite freezes the canonical
  episode-wide SharedObs mode and actor-input projection plus the exact
  configured-roster availability topology, while a custom suite retains its
  explicitly versioned generic provenance;
- critic-information regime, canonical reward mode, and shaping configuration;
- primary, secondary, exploratory, and diagnostic metric IDs;
- independently approved content-addressed layouts/maps and authored initial
  conditions,
  scenarios, rosters/compositions, cooperative partners, adversarial
  opponents, sides, roles, and their joint target weights;
- task/scenario configuration and static-mechanics catalog versions;
- completion, retry, censoring, artifact-retention, and replay policies; and
- the population to which results are intended to generalize.

The official canonical Team Deathmatch suite freezes the mirrored 5v5 roster
with exactly one Mage, Warrior, Hunter, Rogue, and Priest per team, approved
canonical evaluation maps, evaluation seed schedule, cooperative partners and
adversarial opponents, side assignments, canonical reward, metric set, cell
weights, and all other resolved task and lifecycle constants. A custom suite
may freeze any structurally valid roster, including unequal team sizes,
asymmetric compositions, and duplicate classes, but its suite identity and
every report must label it `custom`. Reusing canonical mechanics or even the
canonical roster does not make the suite official or establish that its full
population is the canonical condition.

### Experiment manifest

An experiment manifest fixes the study's choices before locked-test evaluation:

- algorithms, policy architectures, training configurations, and code/artifact
  revisions;
- each official learned checkpoint's canonical SharedObs actor-input projection
  version and compatible compiled actor-front-end contract; any custom
  checkpoint retains its explicit generic mode/projection identity;
- independent training-run identities and named training seeds;
- evaluation seed schedule and common-condition pairing;
- checkpoint-selection rule and selection split;
- train, development, validation, and locked-test partitions;
- comparison estimands, effect measures, uncertainty methods, confidence level,
  and multiplicity families;
- independent-seed budget and precision or stopping rule; and
- hardware, batch size, horizon, JIT/warm-up, repetition, and timing rules.

The manifest is immutable once locked-test evaluation begins. A corrected run
uses a new manifest revision and records the reason.

### Metric result

A metric-result row, conceptually `MetricResultV1`, carries:

- metric, suite, and manifest IDs, versions, and digests;
- training-run and evaluation-seed identities;
- complete evaluation-cell and subject coordinates;
- raw sufficient components and source-schema versions;
- computed value or `null`;
- artifact validity, rollout completion, processing status, and end/failure
  reason;
- a per-statistic endpoint-observation status when the metric has a scientific
  event endpoint; and
- one result status: `defined`, `zero_opportunity`,
  `structurally_inapplicable`, `ambiguous_attribution`, `invalid_artifact`, or
  `insufficient_data`.

`null` is never silently converted to zero. Presentation code may render it as
`N/A` together with the status and reason.

When multiple conditions apply, result status uses this precedence:
`invalid_artifact`, `structurally_inapplicable`, `ambiguous_attribution`,
`insufficient_data`, `zero_opportunity`, then `defined`. In particular, an
ineligible prefix with an observed zero denominator is `insufficient_data`,
not `zero_opportunity`; zero opportunity is available only after artifact,
subject, completion, and endpoint eligibility pass.

The historical M6 CP3 host path implements the generic record contract without
a universal metric registry. When enabled, `EvaluationEpisodeObserverV1` gives
each reducer one validated set: the episode context, frame before the step,
transition, and frame after the step. A reducer returns new state from those
facts. It must be deterministic, leave prior state unchanged, and use recorded
facts rather than recreate simulator rules. `EvaluationMetricReportV1` stores
the episode context once. Its statistic rows refer back to that context, so
policy, seed, task, scenario, information, reward, and shaping records are not
copied onto every row. New scalar TDM runs use the numerical path described in
the [current metric contract](metric_specification.md#current-scalar-tdm-contract).

The observer counts validated transitions separately from processed transitions.
If append or reduction fails, the observer cannot continue as though it
succeeded. It keeps the last validated consecutive prefix and cannot publish
only some of the updated statistics. `EvaluationProcessingStatusV1` reports that
failure independently of `EvaluationEpisodeCompletionV1`; a processing failure
after a completed rollout never rewrites the rollout as failed. There is no
logging sink, file writer, replay envelope, runner framework, or core callback
in CP3.

## Episode, training, evaluation, and scenario ownership

An episode configuration gives the fixed simulator inputs for one rollout.
Training chooses it from a declared distribution; evaluation or a scenario
fixes it in advance. The configuration alone does not say which training or
evaluation population it belongs to. [Amendment A15](../design/specification_amendments.md#a15-episode-configuration-and-experiment-distribution-ownership)
defines structural validity and the permissive Team Deathmatch roster
contract.

Milestone 9 owns training selection. The approved direct Team Deathmatch
training contract uses the canonical mirrored five-class 5v5 roster and the
same task mechanics, lifecycle rules, score threshold `K`, horizon `H`, and
canonical reward as official evaluation. It draws maps uniformly from the
verified training set, maps 0–41. Maps 42–46 belong to validation; maps 47–51
belong to the locked test set. Content identity and declared use enforce these
boundaries; changing a name or map number does not make held-out content
eligible for training. Researchers may define other distributions over
structurally valid episode configurations, subject to the experiment's
declared content-separation rules.

The approved benchmark 1v1–5v5 roster rule uses equal team sizes. Each team
draws its own class subset independently and uniformly, without duplicate
classes. A 1v1 draw excludes Priest. Draws at 2v2–4v4 allow all five classes.
Selected classes fill each team's first slots in canonical relative order;
the remaining slots are inactive. At 5v5, both teams use exactly Mage, Warrior,
Hunter, Rogue, Priest in that order. This rule replaces the earlier handpicked
roster proposal, as recorded in
[A39](../design/specification_amendments.md#a39-sampled-training-maps-and-rosters).
It does not restrict the simulator's wider roster support.

Maps and rosters may change only when an episode resets. A continuing game
keeps its configuration and recurrent memory. An optional curriculum selects
the team size and eligible training maps over time. The implemented training
collector composes that schedule with exact whole-batch budgets, optional team
score shaping and same-run opponent history under
[A40](../design/specification_amendments.md#a40-training-collection-boundaries).
It performs no optimizer update and is not a complete trainer.

Every accounting round advances each lane once; the total transition budget
must divide by the positive even batch size. Requested stage boundaries do not
replace games already in progress. Report their requested budgets separately
from the distributions actually played, including games started, transitions,
active living Team A decisions and unfinished games. Actor decisions are not a
claim about learner sample use. Keep K20/H300 fixed in these four starting
settings: Plain, curriculum only, shaping only, and curriculum plus shaping.

Potential shaping uses the learner's exact discount and the producing game's
team score difference. True wins, losses and horizon draws cancel the next
potential; collection/stage cutoffs and actor death do not. Keep these training
adjustments separate from canonical task rewards, scores and saved metrics.
Shaping grants actors no extra information and proves no learning benefit.

Opponent assignments occur at reset. With stored history, a new game chooses
current weights with probability 0.8, otherwise one stored snapshot uniformly.
Empty history means current self-play. Historical variables stay fixed for
that game. Current variables refresh after a completed learner update, before
the next collection block; each team's recurrent memory stays separate.
Requested snapshot thresholds and actual capture updates/experience must both
be recorded. Several thresholds crossed by one update share one snapshot.
Assignment probabilities do not fix the realized proportion of transitions.

The [training guide](../training/README.md#collect-an-exact-experience-budget)
defines the callable workflow, returned data, padding and recording boundary.
These execution contracts do not establish useful stage exposure, sample
efficiency, learned teamwork or the one-GPU/one-day competence claim.

An evaluation suite fixes its configuration population and policy assignments.
A scenario fixes its episode configuration, slot-by-slot roster, initial state,
roles, horizon, matched seeds, and measured endpoints. The schedule is a stable multi-attempt definition;
each episode's realized seed record joins to exactly one declared schedule
coordinate. Binding policy A to Team A and policy B to Team B, or binding
policies per slot, fills the scenario's frozen roles; it cannot replace the
roster or any other scenario-owned condition. The owning evaluation definition,
not the saved DevClient scenario, binds the behavioral claim, the compared full
method and matched ablation, and any deterministic pressure-controller
protocol. Saved scenarios remain controller-independent.

A saved DevClient scene is an authoring input. To use it as evaluation evidence,
the evaluation owner must load one exact saved revision, normalize and validate
it, and approve it. The owner then binds the layout and authored initial
condition by content digest to the resolved configuration and actual initial
state. Those scientific lifecycle
steps belong to the owning suite, scenario, or manifest rather than to the
DevClient. A filename, moving latest-revision alias, mutable browser buffer, or
successful preview is not an admissible identity or validity proof.

Historical `ResolvedScenarioSpecificationV1` does not bind an explicit
fixed-slot roster, exact role template, or matched schedule identity and
membership rule. Its resolved-config digest contains no roster rows, and
`EvaluationSeedProtocolV1` represents one realized episode rather than the
schedule. M7 C2 therefore introduced the V2 contract, which validates exact
roster/role joins to `EvaluationEpisodeContextV1`, proves that each realized
seed record occupies exactly one declared schedule coordinate, and enforces the
core structural and product config invariants on loaded context. V2 also
freezes and joins the independently approved content-addressed layout and
authored initial condition independently from the resolved-config digest. It
rejects any policy binding that changes the layout, initial condition, resolved
configuration, roster, configured-active slots, role template, schedule, or
realized coordinate. Scenario identity plus a subset check over role names is
insufficient.

The M7 V2 companion records three separate things: the authored content's
identity, the prepared initial state, and the evidence for one recorded attempt. The specification digest binds the
independently approved, content-addressed authored-initial-condition identity
to a stable digest of simulator step count plus the dynamic global snapshot.
Each record separately binds the complete frame-zero digest, including attempt
identity and derived policy
material. This prevents an attempt-specific frame hash from masquerading as the
stable authored-state identity while still proving the exact captured frame.

No training distribution is inferred from an evaluation suite or scenario.
Accepted scenarios are public evaluation instruments; **locked** means that
their protocol is frozen, not that their contents are secret. The protected
scenario closure includes the embedded map, initial state, roster and episode
configuration, pressure controller, seed schedule, endpoints, retained
replays, and result feedback. None of it may influence any adaptive decision,
including policy updates, curriculum or opponent adaptation, reward or
heuristic design, architecture or hyperparameter choices, prompts or decoding,
checkpoint selection, early stopping, repeated-submission selection, or
population weighting. A map embedded in an official scenario is consequently
ineligible for official training and validation populations.

## Common policy and evaluation lifecycle

[Amendment A12](../design/specification_amendments.md#a12-common-milestone-1012-policy-pipeline-spine)
requires shared episode and policy contracts across the current Milestones
8–10 (historically 10–12, under A36). Direct training, custom training,
curricula, evaluation suites, and scenarios all use the same versioned policy
assignment and seed rules. Official execution uses
SharedObs throughout the same reset, base-observation, exact-mask,
legal-action-realization, joint-action-assembly, core-step, transition,
completion, capture/replay, and metric lifecycle. Evaluation/scenario episodes
use M8's host adapter; M9-selected training episodes use M10's JAX
rollout/update/checkpoint adapter. Training selection does not create a runner
or trainer, and training does not route through the evaluation/scenario host
adapter.

[Amendment A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution)
supersedes the earlier forward dual-regime benchmark plan. Official baseline
training and evaluation, controlled scenarios, tournament and cross-play
cells, ratings, leaderboards, and Paper 1 reports require the current relative
input projection introduced by A37:

```text
execution_information_mode = shared_obs
actor_projection = base-observation-plus-authorized-sensor-source-bank@2
```

Every official replay frame must carry exactly the availability matrix derived
from its frozen roster: a recipient/source entry is true if and only if both
slots are configured-active, have the same configured team, and are distinct
global slots. The matrix does not depend on living state, health, visibility,
score, policy identity, or frame index. Configured-but-dead teammates remain
authorized sources while their ordinary sensor material remains
lifecycle-zeroed. Exact equality on every frame rejects both an incomplete
stable topology and any mid-episode topology change. An all-false matrix is
valid only when the roster itself contains no same-team off-diagonal pair.

Execution mode, projection, and availability remain mandatory invariant
provenance, not official comparison or aggregation axes. NoSharedObs remains
available through generic rollout and validation for diagnostics, custom
research, and historical replay compatibility, but it cannot produce a new
official baseline, scenario result, tournament cell, or Paper 1 claim. The
current roadmap contains no mixed-regime V2 work. The immutable V1 wire
contract remains homogeneous and dual-mode-compatible; it must never encode
mixed execution by overloading policy identity or availability.

Milestone 8 owns the common host-runner and evidence integration work that is
not superseded by A25. Milestone 9 selects independently approved,
content-addressed partition-neutral maps into training populations. Milestone
10 consumes those selections through one SharedObs JAX rollout/update path and
owns compatible learned actor front ends, checkpoints, and a benchmark
reference encoder. It freezes categorical identities, domains, sentinel
meanings, shapes, masks, slot/source identities, provenance, and
reference-encoder conventions—not researchers' choice of one-hot, embedding,
attention, or other neural representation. M10 rejects held-out
evaluation/scenario assets before training execution.

Centralized training remains separate from actor execution. An explicitly
authorized critic may consume training-only team or privileged information,
but every evaluated actor selects its action from the canonical SharedObs
contract. Critic inputs never become actor availability or evaluation-frame
actor material.

[Amendment A30](../design/specification_amendments.md#a30-reactive-tdm-and-specialist-scenario-controllers)
retired the legacy Scripted TDM/MRP executables. Reactive TDM supports either team.
[Amendment A31](../design/specification_amendments.md#a31-scenario-5-priest-pursuit-controller)
introduced Team B's Scenario 5 Controller. Its
[A32 behavior v2](../design/specification_amendments.md#a32-scenario-5-shoulder-bypass-and-fallback-prey)
keeps Reactive TDM for other classes; Rogues pursue observed living Priests
first, otherwise Hunters, with glancing shoulder contact and independent legal
combat. Under
[A33](../design/specification_amendments.md#a33-one-controller-for-scenarios-3-and-5),
both scenarios adopted behavior v2 through the single former
**Scenario 3 and 5 Controller** option. Under
[A34](../design/specification_amendments.md#a34-reactive-tdm-alpha-and-beta),
the existing general controller is displayed as **Reactive TDM ALPHA** and
the shared variant as **Reactive TDM BETA**. BETA behavior v3 pursues Priest,
then Mage, then Hunter; other classes and no-prey Rogue movement remain ALPHA.
Rogue combat remains independent of pursuit and bounded by its own Basic radius
and exact masks. The internal `scenario_5` identity and
V5 payload remain unchanged; the old Scenario 3 executable is removed while
historical evidence retains its original identity and outcomes. Earlier Scenario
3 success claims must be rechecked against the new opponent, not relabelled.
All current reactive controllers require SharedObs and remain diagnostic or scenario-pressure
tooling, not official baselines. Random remains diagnostic quality-control
tooling. Under
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations),
an official scenario evaluation independently freezes and binds the selected
deterministic controller through `pressure_protocol`, identically for the full
method and matched ablation. Merely selecting a controller in the DevClient or
passing its physical regression does not qualify official scenario evidence.
[Amendment A27](../design/specification_amendments.md#a27-rolling-big-12-and-baseline-library-governance)
records the earlier rolling Big 12 and cumulative Baseline Library direction.
The accepted monthly snapshot, optional challenger, and admission rules in
[Big 12 tournament and Baseline Library](#big-12-tournament-and-baseline-library)
supersede its earlier weekly/fixed-budget details. These governance changes do
not alter the simulator, generic cross-play rules, or the slot-based focal,
partner, and opponent roles.

## Experimental units and terminology

- A **training run** is one independently initialized and trained policy or
  policy set. It is the default experimental unit for algorithm-level claims.
- A **training seed** is provenance for a training run, not a guarantee of
  independence when weights, replay, curriculum state, or selected checkpoints
  are shared.
- An **evaluation episode** is one rollout of a frozen checkpoint under one
  evaluation condition and seed.
- An **evaluation cell** is one homogeneous condition in the Cartesian product
  below.
- Episodes, transitions, agents, classes, deaths, objectives, and the two teams
  in one match are repeated or nested observations. They are not independent
  algorithm replicates.

The canonical cell coordinates are:

```text
battleground identity/version x layout/map
     x cooperative-partner/pool x adversarial-opponent/pool
     x side/role x roster/composition
```

Use an explicit not-applicable value when there is no separate partner or
opponent. Keep partner and opponent as separate coordinates. If both change
at once, a score change cannot by itself tell us whether partner cooperation
or opponent difficulty caused it.

Resolved episode configuration, scenario version, policy IDs,
critic-information regime, reward mode, shaping configuration, code revision,
static-catalog digest, execution mode, actor-input projection version, and
availability remain mandatory provenance. Canonical actor-information fields
are constant eligibility facts and do not become official cell axes. Raw
task-score units are never pooled across battlegrounds. A cross-battleground
scalar requires a separately versioned normalization and suite contract.

## Suite construction and frozen weights

Each suite declares a finite set of cells and nonnegative target weights
`w_c` summing to one within each reported task. Equal cell weights are the
default. A task-declared target distribution is allowed when it is motivated
and frozen before evaluation.

Weights describe the intended population. They do not depend on how many
episodes finish. If a cell fails, has no opportunities, or lacks an artifact,
keep its declared weight. Apply the suite's missingness rule and report coverage;
do not silently increase the weights of the remaining cells.

Layouts, sides, rosters, opponents, partners, and scenarios must not be sampled
from an undocumented changing distribution. Joint partner/opponent weights are
frozen; marginal weights are insufficient when the sampling design is not a
Cartesian product. If procedural generation is used, the generator version,
parameter distribution, and seed schedule are part of the suite.

## Aggregation

### Episode scalars and outcome distributions

For an eligible episode scalar `x`, training run `r`, cell `c`, and eligible
episode count `E_rc`, first compute:

```text
x_bar[r,c] = sum_e x[r,c,e] / E_rc
```

The run-level target-population estimate is:

```text
x_run[r] = sum_c w[c] * x_bar[r,c]
```

Win/draw/loss is one multinomial endpoint. Apply the same operation to the
three mutually exclusive indicator components; do not present three unrelated
binomial experiments.

For Team Deathmatch, only reaching the configured score threshold can author a
win. If neither team reaches it by the horizon, the task authors a draw even
when terminal scores differ. A simultaneous double crossing compares complete
successor scores, with equality drawing. Reports preserve terminal score
differential as descriptive evidence and preserve the completion basis as
`score_threshold`, `horizon`, or `score_threshold_at_horizon`; they never infer
the result from score differential, reward, or done flags.

### Opportunity rates and shares

For a numerator `n` and genuine-opportunity denominator `d`, average the raw
components inside each run/cell:

```text
n_bar[r,c] = sum_e n[r,c,e] / E_rc
d_bar[r,c] = sum_e d[r,c,e] / E_rc
```

Then compute the run-level rate as the ratio of weighted components:

```text
q_run[r] = (sum_c w[c] * n_bar[r,c])
           / (sum_c w[c] * d_bar[r,c])
```

Use the weighted raw components above, rather than an average of cell rates.
Keep independent training runs separate and keep the declared cell weights,
including cells with zero opportunities. Save every `n_bar`, `d_bar`, and defined
cell rate so another reader can check the calculation.

If the weighted denominator is zero for a training run, the run result is
`zero_opportunity`, not zero. Report the incidence of zero-opportunity runs.
A summary over only defined runs is explicitly conditional and is not an
unqualified confirmatory estimate. A confirmatory conditional-quality claim
therefore reports two parts: opportunity exposure and quality conditional on
that exposure.

Agent/class shares preserve both the subject component and team total. A
structurally absent class is `structurally_inapplicable`; a present capable
agent with zero output has a defined zero volume. A share is
`zero_opportunity` when its team total is zero.

### Durations and distributions

For time-based rates, keep both qualifying and eligible agent-steps. Store
event and episode observations with their cell and training-run IDs. Use the
median and interquartile range by default; use tail quantiles when the research
question needs them. A minimum or maximum alone does not establish robustness
or quality.

### Across independent training runs

Independent training runs receive equal weight. Report the run-level values,
the aggregate estimator, uncertainty across runs, and the number of defined
and non-defined runs. Never obtain artificially narrow intervals by treating
episodes, agents, matches, or policy pairings as independent seeds.

Common evaluation seeds and side swaps reduce rollout noise, but do not make
independently trained runs a paired experiment by themselves. Paired inference
is used only when the manifest deliberately pairs the relevant independent
training runs or applies a within-run comparison.

## Inference and endpoint discipline

Every confirmatory comparison predeclares:

- the population contrast and comparator;
- absolute and, when meaningful, standardized or relative effect measures;
- the independent resampling/analysis unit;
- confidence level and interval method;
- one- or two-sided hypothesis direction, if hypothesis testing is used;
- the endpoint family and multiplicity adjustment; and
- how undefined, failed, and insufficient runs affect the claim.

Uncertainty depends on independent training runs, so there is no universal
seed count that makes every claim reliable. Before evaluation, record the seed
budget and the precision target or stopping rule. If there are too few runs
for a defensible uncertainty estimate, label the result descriptive. More
episodes from the same run do not create more independent training seeds.

For multi-task or multi-suite comparisons, report task-level results first.
Interquartile mean, optimality-gap, and performance-profile summaries with
stratified bootstrap uncertainty are appropriate only when normalization and
task sampling are declared. They do not replace the underlying task results.

Only blocks labeled `primary_confirmatory` define the default confirmatory
family. A descriptive coordination block does not become confirmatory merely
because it appears on the compact card. Key secondary endpoints form named
families by scientific claim. A predeclared matched scenario contrast may be
primary for its one bounded behavioral claim, but it is not evidence of general
strength and does not enter Elo. Exploratory, diagnostic, and post-hoc slices
are visibly labeled and cannot be promoted after looking at locked-test
results.

## Selection, splits, and leakage control

Map/layout, opponent, partner, scenario, and seed selections assigned to
training, development, validation, and locked-test partitions are disjoint
wherever the intended claim is generalization. The manifest identifies every
overlap that is intentional.

Map assets themselves are partition-neutral. Training, development,
validation, official/custom evaluation, controlled-scenario, and locked-test
membership belongs to the selecting distribution, suite, scenario, or
experiment manifest rather than to a split field, filename, or directory in
the map document. A manifest may use an exact saved DevClient revision as an
import source, but may reference only the independently approved,
content-addressed resolved contract representation in an evaluation
population. Current browser buffers and moving latest-revision aliases are
ineligible.

An official evaluation suite or scenario is never a default training
distribution. Public availability does not authorize reuse for learning. The
complete protocol-frozen scenario closure—embedded map, initial state,
roster/configuration, pressure controller, seeds, endpoints, replays, and result
feedback—must not influence gradients, online or offline learning, imitation,
behavioral cloning, distillation, curricula, opponent adaptation, architecture,
hyperparameters, reward or heuristic design, prompts, decoding, checkpoint
selection, early stopping, repeated-submission selection, population weights,
or any other adaptive decision.

Training, validation, and evaluation manifests identify content by immutable
digest and fail closed when their declared content closures intersect where
disjointness is required. They never resolve a mutable latest-revision or
`latest_big_12` alias. M9/M10 owns enforcement before training begins; full
training, checkpoint, selection, and population provenance must remain
available for maintainer reproduction.

- Training data may update policies and curricula.
- Development data may debug implementations and metric code.
- Validation data may select checkpoints, tune hyperparameters, and validate
  predeclared analysis parameters, provided that data is disjoint from the
  official scenario closure.
- Locked-test data may evaluate a frozen decision only. It may not choose the
  metric, formula, threshold, opponent pool, checkpoint, reward shaping, or
  presentation cutoff.

The checkpoint-selection rule must be executable from validation information
alone and must state tie-breaking. Best-of-many test selection is prohibited.
When a locked-test defect requires rerunning, preserve the failed attempt,
explain the defect, and revise the manifest rather than overwriting history.

Intentional undisclosed use of official scenario material for adaptation is
scientific misconduct. Open-source software cannot make that conduct
universally impossible, so this benchmark uses proportionate controls:
content-addressed manifest separation, complete provenance, and maintainer
reproduction. Failure to reproduce makes a result ineligible under the frozen
protocol, but non-reproduction alone is not proof of fraud.

## Completion, failure, missingness, and censoring

Check four things separately: whether the artifact is valid, whether metric
processing succeeded, whether the rollout finished, and whether each scientific
endpoint was observed. Success or failure in one does not answer the others.

### Artifact validity

A valid artifact uses recognized schema/catalog versions, matching digests,
consecutive indices, and consistent identities. Arrays have the required shapes
and finite values. Catalog mappings cover every fixed slot and agree with the
episode roster. Completion fields must agree with one another. Report invalid
artifacts as failures or quality-control evidence; do not use them for tactical
metrics.

### Rollout completion

- **Complete:** the task emitted an authoritative terminal outcome or the
  declared evaluation horizon was reached. The completion basis remains
  explicit. Truncation is preserved separately and does not by itself make a
  task outcome observed. Team Deathmatch horizon completion is an explicit
  authoritative draw, not an outcome inferred from truncation.
- **Partial:** an intentional stop preserved a valid gap-free prefix without
  completing the declared rollout.
- **Interrupted:** an external interruption preserved a valid gap-free prefix.
- **Failed:** simulation, policy, validation, or capture failure prevented the
  intended rollout. Its stable failure origin and reason remain visible under
  the manifest's predeclared policy.

The initial frame is required before any completion record exists. A
zero-transition episode may therefore be partial, interrupted, or failed after
valid frame zero, but it cannot be complete. Artifact frame zero is independent
of simulator epoch zero: a valid capture may begin at any nonnegative simulator
step. A complete rollout has either authoritative task termination or exactly
the declared number of artifact transitions; no host consumer derives task
outcome from that structural evidence.

### Observer processing

Processing success means that every validated transition was consumed by every
declared reducer and the final report was assembled atomically. Trusted pure
reducers own valid immutable outputs; cheap identity, progress, eligibility,
uniqueness and provenance checks protect the computation boundary. Repeated
deep validation of computed state/report history is not part of metric
computation. External artifact ingestion and independent tests retain full
record validation. Processing
failure records the stage, stable code, stage-governed reducer identity and
attempted-transition provenance, and diagnostic detail. Reducer initialize,
advance, and finalize failures require the exact reducer identity/version;
non-reducer stages forbid one. Reducer advance requires an attempted transition
index equal to processed progress and leaves validated progress exactly one
unit ahead. Transition validation may preserve the submitted attempted index as
diagnostic provenance even when it is malformed or noncontiguous. Other stages
forbid an attempted index. These rules do not erase the last validated prefix
or alter already-authored termination/truncation truth. A report exposes both
validated and processed counts, and no successful report may contain a partial
subset of reducers or rows. A failure after every validated unit was processed
remains visible but does not by itself make a complete-only statistic
ineligible; a validated/processed count gap does.

Retry count, retryable causes, replacement-seed policy, and maximum attempts
are frozen. A retry never silently erases its failed predecessor.

### Scientific censoring

Censoring means the record is valid but the scientific observation window ended
before the endpoint was seen. It does not mean the artifact is corrupt, and it
does not replace the episode's completion state.
Each statistic records one endpoint-observation status from `not_applicable`,
`observed`, `right_censored`, `competing_event`, or `unavailable`. For
time-to-success:

- success by the horizon is an observed event;
- horizon reached without success is a failure for binary success-rate
  estimation and right-censored for time-to-success;
- a task-terminal policy failure is a competing failure, not successful
  completion;
- infrastructure interruption is invalid/interrupted data, not scientific
  censoring; and
- completion time among successes is never reported without success
  probability and censoring information.

Different statistics from the same complete rollout may legitimately have
different endpoint-observation statuses. A prefix-valid descriptive statistic
may remain defined on a partial rollout while a complete-only outcome is
`insufficient_data`; neither case turns missing future opportunity into zero.

The missingness rule must say which population the claim describes: all
scheduled runs, successfully trained runs, or valid completed evaluations.
Choose that population before seeing failures.

## Symmetry, sides, and common conditions

Both teams use the same metric definitions. Suites use side swaps and paired
layouts/seeds when the task permits them. Side effects are reported and remain
part of the cell coordinates. The two sides of one match are paired views of a
single stochastic event.

For self-play or policy-versus-policy evaluation, policy identity and side are
not conflated. A symmetric matrix records both assignment directions or uses a
task-justified symmetry reduction that remains recoverable.

## Compact scorecard reporting

The [metric specification](metric_specification.md#presentation-budgets)
exclusively owns the contents and budgets of the primary team and agent/class
cards. An evaluation suite instantiates that scorecard for each task under the
canonical SharedObs eligibility contract. This protocol does not redefine the
scorecard's metric blocks. The suite and manifest freeze which eligible
candidate fills each optional slot and place those primary blocks in the
confirmatory endpoint family before locked-test evaluation.

Every primary table footer reports suite/manifest versions, independent
training runs, cells, episode schedule, defined/undefined counts, failure and
truncation rates, cell weighting, and uncertainty method. Raw sufficient
components remain exportable.

## Controlled-scenario protocol

Use controlled scenarios to test a tactical claim whose meaning depends on
context. Compare a full method with one declared ablation under the same fixed
conditions, and measure the claimed behavior directly. Each versioned scenario
evaluation fixes:

- one bounded behavioral hypothesis and the eligible policy roles;
- the full-method and matched-ablation identities;
- independently approved content-addressed layout and authored-initial-condition
  identities;
- its resolved episode configuration and exact fixed-slot roster;
- initial state, role template, horizon, and matched seed schedule;
- one deterministic, content-addressed pressure-controller protocol;
- one primary quantitative endpoint;
- at most two supporting secondary margins;
- explicit safety, role, or behavior violations;
- success, failure, terminal, and censoring semantics; and
- replay retention and blinded review sampling.

The full method and ablation must use the same scenario revision, embedded map
and initial state, pressure controller, canonical SharedObs contract,
evaluation seeds and side assignments, training budget, checkpoint-selection
rule, and primary endpoint. Differences outside the declared ablation invalidate
the contrast.

Policy assignment is an execution binding, not scenario construction. A
Team A/Team B convenience binding and a lower-level per-slot binding must both
respect the scenario's fixed roster, configured-active slots, and role
template. Changing those facts creates a different versioned scenario rather
than an override of the current one.

Pressure controllers are evaluation fixtures, not general policies or action
tapes. Each controller reacts deterministically to the current same-epoch
canonical SharedObs inputs and authoritative action mask. Its versioned contract
names the controlled slots, target-selection rule, deterministic tie-breaking,
and legal fallback when an intended action is unavailable. The identical
controller version and seed binding apply to every treatment/ablation policy in
the comparison. Scenario-specific behavior is selected through the resolved
`pressure_protocol`; it is never implemented by branching the generic TDM
controller on a scenario name or ID.

Every official scenario attempt binds canonical SharedObs after loading and
validating that regime-independent scenario. Its mode, projection, and exact
all-frame configured-roster availability must pass the official replay-level
gate before the result is accepted.

Scenario validation must also join the explicit fixed-slot roster and role
template carried by the resolved scenario definition to the episode context,
then prove that the episode's realized seed record is exactly one member of the
scenario's stable matched schedule. A set/subset comparison of role labels or
a digest of one realized seed record does not establish those identities.

“Single-shot” means no learning or adaptation across attempts. It does not mean
one episode is enough for a stochastic method. Compare policies using the same
fixed scenario seeds when possible.

The approved TDM suite contains exactly eight scenarios. Scenarios 1, 2 and
4–8 have five-transition horizons; approved Scenario 3 r24 alone has a
ten-transition horizon for sustained body blocking, with the canonical
five-transition respawn-wave period. Scenario/map design approval is complete
under A36; remaining identity, transport and evidence checks do not reopen it.
These horizons are properties of this suite, not a global scenario-schema limit.
The packaged Scenario 3 r26 changes only the accepted r24 Notes; its physical
state, map and configuration are identical. The package manifest retains both
the approved source and distributed revision identities.

`marl_battlegrounds.tasks` provides public map/scenario discovery and loading,
`make_standard_team_deathmatch_config` for explicit approved maps and independently
ordered rosters, and `make_canonical_team_deathmatch_evaluation_config` for the
five canonical evaluation maps
(47–51) with mirrored Mage/Warrior/Hunter/Rogue/Priest slots.
These constructors select immutable packaged content without importing the
mutable DevClient draft store. Canonical evaluation retains paired side
assignments as well as the approved geometric symmetry.

The [2026-09-16 map publication approval](../design/specification_amendments.md#approved-map-publication-on-2026-09-16)
selects audited revisions for new games: 39 physical map updates, Map 3's
obstacle-ID cleanup and 12 unchanged maps. A36 owns the exact clearance limits
and exceptions. Publication and focused replay compatibility checks passed.
The packaged manifest records the exact selected sources; prior identities
remain in one immutable history resource so old recordings keep their original
map version. New runs use the current catalog. Restart long-running clients
and Python processes after the package update to refresh cached map records.
The eight scenarios carry their own saved configurations and winning sequences;
this map publication does not replace them. Earlier solver results, timings and
replays describe their recorded map versions, not the revised layouts.

`python -m scripts.dev.qualify_tdm_scenarios <new-directory>` exercises all
eight packaged definitions at two fixed schedule coordinates through the shared
`evaluate_episodes` executor, current scalar metrics, replay V3 and scenario
record V4. One RunWriter run retains one full scalar CSV row per episode. Team A
uses ALPHA as a pipeline control; Team B uses the approved ALPHA/BETA pressure
binding. Before capture, the context binds the actual built-in callable's
versioned controller descriptor separately from its frozen variables digest,
together with the approved scenario identity, fixed roles and public slot IDs.
The recorded RNG protocol and root/episode coordinates are those actually
consumed by the shared evaluator; derived named streams remain unrecorded.
The suite index retains source/configuration/controller identities, completion,
quantitative terminal Team A reward and every failure. A retained partial prefix
has no completed scalar row and no available terminal endpoint. This establishes
reproducible integration; learned-policy, matched-ablation and manuscript
campaigns belong to later milestones. It introduces no new scenario approval gate.

Each treatment/control arm uses multiple independently trained, deliberately
paired runs. The training run is the replication unit; agents, ticks, episodes,
and the two teams within an episode are nested observations, not independent
replicates. A result supports the predeclared behavior under the frozen
scenario conditions. It does not prove a universal policy trait, contribute to
Elo, select a checkpoint, or authorize the scenario endpoint as a curriculum or
shaping objective.

Peeling, kiting, flanking, body blocking, triage, regrouping, rotations,
escort/interception, Trap discipline, Burst synchronization, and
Freedom-assisted movement use this protocol. A scenario may demonstrate
behavior under its frozen conditions; it does not prove a universal policy
trait beyond them.

Generic teamfight or engagement segmentation is outside this protocol. The
suite does not plan a detector, validator, or teamfight-conditioned endpoint;
authoritative task context and controlled scenarios own contextual tactical
claims.

## Cross-play and population evaluation

Cross-play measures a task endpoint across a fixed set of policy combinations.
Keep cooperative partners separate from adversarial opponents. Cross-play does
not need a separate copy of every tactical metric.

Every official cross-play episode uses canonical SharedObs. Information regime
is therefore constant provenance, not another tensor axis or assignment
direction. A NoSharedObs or mixed-regime matchup is custom diagnostic evidence
and cannot enter the official panel, aggregate, or rating calculation.

Preserve and publish the complete
`focal policy × cooperative partner × adversarial opponent × side assignment`
tensor whenever feasible. A smaller design must be a predeclared balanced or
inclusion-weighted subsample with recoverable inclusion probabilities, frozen
joint weights, and both task-relevant side assignments. A two-dimensional
“partner/opponent” matrix is insufficient when both roles vary.

At minimum, declare:

- cooperative-partner and adversarial-opponent identities and not-applicable
  semantics;
- matched/training-related and held-out/disjoint pool definitions;
- pool construction, policy checkpoints, source training runs, and weights;
- matched-partner and held-out-partner performance under the same frozen
  opponent distribution;
- held-out-opponent performance under the same frozen partner distribution;
- a partner-generalization gap accompanied by both absolute components;
- a predeclared lower-tail statistic and sensitivity for one explicitly named
  population axis; and
- whether the claim targets a fixed official panel or a broader population.

Raw worst-partner minimum is not the primary robustness statistic because it
is pool-size and outlier sensitive. A cross-play gap is always accompanied by
both absolute components; a small gap caused by uniformly poor performance is
not robustness.

For a fixed official panel, population members are fixed evaluation cells and
uncertainty is across focal independent training runs. For a
population-generalization claim, focal, partner-source, and opponent-source
training-run identities are three crossed dependence dimensions and must be
resampled or modeled accordingly. Reusing one partner or opponent policy in
many tensor cells does not create independent population samples. Matched
versus held-out partner contrasts hold the opponent distribution and its joint
weights fixed; opponent contrasts analogously hold the partner distribution
fixed.

Matched-partner metrics are `structurally_inapplicable` for a monolithic
full-team policy without separable partner assignments.

## Big 12 tournament and Baseline Library

**Canonical execution, reuse and separate maintainer admission machinery are
implemented. Official rules, trained controllers and a qualified released bundle
are still required.** A released Big 12 snapshot contains exactly
twelve method-level entrants. Each has one fixed executable system and one Elo
value. An ordinary canonical call evaluates those twelve alone, or adds one
challenger to make thirteen. A system may contain several internal policies;
those components do not become extra entrants. Larger or edited populations
use the custom tournament route.

The tentative Paper 1 roster remains:

1. RNN-IPPO with parameter sharing.
2. RNN-MAPPO with parameter sharing.
3. RNN-MAPPO with class-specific actors.
4. RNN-HAPPO.
5. HyperMARL-PPO.
6. RNN-QMIX.
7. RNN-PQN-VDN.
8. MAPPO-PFSP League.
9. MAPPO-PSRO.
10. `S*-Curriculum`.
11. `S*-Curriculum-Shaped`.
12. Qwen-Five.

For rows 1–11, retain three independently initialized training runs and their
eligible checkpoint histories. A rule fixed before training selects one
checkpoint from each run using validation only, then selects the best eligible
validation checkpoint of the three as that method's tournament system. Record
the validation metric, cadence, eligibility, and every tie-break rule in the
manifest. Scenario or tournament results cannot select the checkpoint. Keep
all training-run and selection evidence even though only one system enters.

Qwen-Five remains tentative. It must pass measured throughput, latency,
resource, reproducibility, and protocol-compatibility checks. It is not assumed
to fit a JAX training loop. If it fails, its place stays unresolved until an
explicit governance decision chooses what happens next.

An immutable monthly configuration names the exact twelve controller
versions, approved maps and mirrored rosters, SharedObs input contract, game
budget, seed schedule, memory/rating settings, and artifact references with
hashes and sizes. Resolve it once. Resume must use the saved resolved version;
a newer installed default must not change an existing run. Missing official
assets must produce a clear error, never substitute ALPHA, BETA, or another
library member. Models and reports remain separate from the small configuration.

The resolved field has 66 unordered matchups for twelve entrants and 78 for
thirteen. Every matchup uses the same game budget, equal coverage across the
five test maps, and complete default/swapped spawn pairs. The challenger stays
Team A; incumbent games keep their recorded A/B assignments. Exchange complete
ordered spawn banks, including respawns. Keep roster order, world directions,
action meanings, map/seed pairing, and fresh per-game memory fixed.

**The official numerical game budget is not yet approved.** An omitted
`games_per_opponent` inherits the snapshot's positive integer budget. An
explicit budget must be positive, nonboolean, and divisible by ten for five
maps. It applies to every incumbent and challenger matchup. A different budget
is a declared research override and cannot qualify promotion; an explicit equal
budget remains compatible with the official rule. These argument rules are
implemented; they do not settle the official numerical budget.

By default, the canonical runner verifies and reuses all 66 incumbent matchups.
With a challenger, it runs twelve more matchups and fits all thirteen systems
together. Reuse raw games, never old Elo as rating credit. Check controller,
protocol, environment/schema, map, roster, assignment, spawn, budget, and random
stream identities before reuse. Preserve original game IDs and seeds when an
entrant is added or reordered. Required outcome and requested metric/replay
coverage must be present before starting new games.

Select any smaller reused set in the snapshot's declared order without looking
at outcomes. Keep whole spawn pairs and larger declared seed groups together.
If compatible coverage is insufficient, reject and explain the explicit
`rerun_existing=True` route. That route runs the entire resolved tournament
again; it does not silently fill only missing games. Report planned, reused,
and new game counts. Incomplete or failed runs keep valid raw records but cannot
claim a finished tournament or headline result.

**Implemented custom behavior:** `run_tournament` accepts an explicit System or
Policy population, or an immutable configuration. Its programmatic default is
`episodes_per_pair=100`, with fixed Team A/B ownership and opposite complete
spawn banks. There is no twelve-entry limit on custom fields. Canonical calls
use the same execution, statistics and result authorities, with verified reuse
and one optional challenger. See [Canonical Tournaments](canonical_tournaments.md).
A qualified official bundle and its numerical budget remain separate release
gates. Historical team-swapped records keep their original meaning. The earlier
6,600-game workload had ten seed blocks per map with opposite policy-side
assignments; that historical workload is not an official game budget.

The compact report is **Policy | Elo with uncertainty | Expected Score with
uncertainty | Win % | Draw % | Loss %**, with the matchup matrix and per-map
breakdowns available beside it. Outcomes are always required. Tactical metrics
and replays are independent choices and are not needed to fit ratings. A
finished canonical result requires all scheduled outcomes; it must not silently
drop a failed matchup.

### Frozen rating and uncertainty contract

Let `s` be centered log strengths with `sum(s) = 0`, and `z` the log draw
parameter. For policies i/j use logits `[(s_i-s_j)/2, z, -(s_i-s_j)/2]` for
win/draw/loss. Thus draw weight is `exp(z)` without an extra factor of two.
Fit summed negative log likelihood plus
`(sum(s²) + z²) / (2 ln(10)²)`. Elo is `1200 + 400s / ln(10)`; a rating difference
describes decisive win/loss odds, with draws modelled separately. The symmetric
regularizer stabilizes separated/constant results; it does not create new evidence.

Use existing BFGS with a fixed absolute gradient tolerance of `1e-4` and at most
1,000 iterations. Parameters, objective and gradient must be finite and convergence
must succeed for the point fit and every bootstrap replicate. The optimizer may
subtract the parameter-independent saturated-model likelihood to reduce numerical
cancellation; this does not change the objective's minimizer or gradient. Report
the full negative log likelihood plus penalty in estimator metadata.

The implemented rating fit uses the qualified SciPy CPU path once after the
tournament. Historical evidence: a controlled 6,600-match, 5,000-resample
comparison took about 10.5 seconds on CPU with all fits passing; the faster JAX
GPU prototype failed convergence qualification. This is retained estimator
history, not a current CPU speed target or a new qualification result. Numerical
per-episode metrics stay on JAX and can run on GPU. The 1,200 display center
changes neither fitted strengths nor rating gaps.

Use 5,000 deterministic bootstrap replicates, resampling independent seed blocks
within pair/map cells while retaining each complete paired block. Current
schedules preserve fixed-team default/swapped spawn pairs through the versioned
schedule-to-statistics join. Historical team-swapped schedules retain their
original interpretation. Existing records must not be relabelled. Intervals are approximate 95%
percentile intervals conditional on the frozen systems and map panel. They do not
represent training-run variation. Failed fits are errors and are never discarded.
Constant samples and inadequate independent variation produce an explicit
**Insufficient Variation** or insufficient-block result rather than zero-width
certainty.

Joint Elo intervals additionally require a strongly connected directed graph of
observed decisive wins: every group must have some observed reversal path against
its complement. Without it, resampling cannot create missing decisive evidence,
and regularization alone cannot justify uncertainty bounds. Point ratings remain
reportable; expected-score interval availability is assessed separately. This is
a conservative availability rule, not a guarantee of exact nominal coverage near
population boundaries. Report interval availability and the approximation limits.

Expected score uses win=1, draw=0.5 and loss=0, with equal map weighting and declared
opponent weights. W/D/L proportions use those same population weights. Persist
`match_results.csv`, `matchup_results.csv`, `map_results.csv` and
`tournament_results.csv`; the matrix is a view of matchup rows. Secondary lower-tail
robustness is the mean expected score over the worst-performing 20% of weighted
opponent × map cells. For a twelve-system field with five equally weighted
maps, this is the lowest eleven of fifty-five cell means. A thirteen-system
field has sixty such cells and uses its lowest twelve. The same worst-20% rule
applies; it remains outside the compact main report.

### One-day training claim

The predeclared target is a frozen learned policy trained within 24 hours on one
RTX 5090 that beats **both ALPHA and BETA** and has the highest jointly fitted Elo
in their three-way tournament, exclusively on held-out maps. For each baseline,
the lower 95% confidence bound on expected score must exceed 0.5. Confirm the
result across multiple predeclared independent training seeds.

Held-out maps never guide training, hyperparameter choices or checkpoint selection;
validation maps serve those purposes. Freeze the checkpoint before the held-out
tournament. Declare its evaluation budget before seeing outcomes. Include compilation,
rollout, learning and requested logging in the training-time budget; report tournament
qualification time separately. Runtime/VRAM measurements without learner trials do
not establish competence or sample efficiency. Report learning curves, environment
transitions, peak memory, hardware/software identity and the result for every
predeclared training seed, including failures.

The accepted post-Paper-1 process publishes one immutable twelve-system
snapshot at **00:00 Europe/London on the first of each month**. The submission
cutoff is **72 elapsed hours** before that instant. Publish both local and UTC
times; three local calendar dates are not always 72 elapsed hours.

Maintainers process complete, qualified submissions in arrival order against
a provisional Big 12. Each admission freezes that provisional population. The
published twelve stay unchanged until monthly release. Failed or unfinished
work leaves membership unchanged and rolls forward without losing queue order.
A local researcher call neither promotes a challenger nor publishes a snapshot.

Use the complete joint thirteen-system Elo fit for promotion. A challenger
strictly above the lowest incumbent replaces it; an exact cutoff tie does not
promote. After promotion, retain the surviving twelve's 66 matchups: 55 among
old incumbents plus eleven involving the new member. Refit those twelve from
those games before evaluating the next challenger. Do not carry the old Elo
numbers forward as evidence. Apply each admission once and preserve removed
members and their historical games.

Admission requests full metrics from the first transition of every challenger
game. Publication requires complete outcomes, priority data, and full reports
for every retained game; selected replays remain separate. A prior priority-only
run cannot manufacture missing full data. Resource or write failures cannot
become invented wins, losses, or draws to meet a release date.

**Remaining gates:** approve and qualify the official numerical game and
resource limits, timing definitions, the eviction rule when several incumbents
tie for last, revised-method admission/replacement, episode-local adaptation,
and remaining eligibility, reproduction, and public-test-feedback rules. The
monthly cadence and 12-or-13 population are settled design choices; these open
gates still block official launch. Ratings from different snapshot populations
are not directly comparable without a separate model for changes over time.
Paper 1's frozen snapshot and results remain permanent.

The Baseline Library keeps growing as the active Big 12 changes; removing a
member from the active twelve does not remove its historical record. It retains every current and former member's implementation, official
configuration, provenance, designated retained checkpoints, selected final
system, compatibility metadata, and historical membership and results. A
legitimate training workflow may consume compatible library material only
through an immutable manifest declaring exact identities and weights. It must
never resolve a mutable `latest_big_12` population.

Scenario pressure controllers, general Reactive TDM, Random, privileged
policies, intermediate unqualified checkpoints, and PSRO's internal population
members are not Big 12 entrants. Scenario controllers are also outside the
Baseline Library. Qwen artifacts may be retained after admission, but their use
in training remains separately capability- and cost-gated. Maintainers must be
able to reproduce every admitted fixed system under its frozen protocol; an
unreproducible result is ineligible, without that failure alone constituting a
fraud finding.

## Learning and sample efficiency

Measure learning curves at fixed evaluation checkpoints using conditions held
out from training. Validation may guide the declared selection rule; locked-test
results may not. Ordinary curriculum rollouts do not run the full evaluation
suite.

The planned learner contract uses the same canonical SharedObs rollout,
batching, update, and checkpoint lifecycle for direct training, researcher-defined
benchmark distributions, and curricula. The curriculum differs only by
retaining selector state and emitting the next ordinary episode specification.
Roster selection does not create a separate trainer. A benchmark checkpoint is
emitted for the canonical projection/front-end contract, and the learner
interface must accept every structurally valid fixed-slot roster without a
simulator-schema change or a change to the semantic actor-input contract.

Every learning result records:

- environment transitions;
- active-agent decision transitions;
- optimizer updates and samples consumed when applicable;
- wall-clock and compute/hardware provenance;
- evaluation checkpoint schedule; and
- task, roster size, canonical mode/projection, reward mode, and shaping
  identity.

Both environment and active-agent decision transitions are required because a
1v1 and a 5v5 environment transition expose different amounts of agent
experience. Every area-under-curve or time-to-threshold result names its x-axis,
fixed horizon, interpolation convention, threshold, and treatment of runs that
never reach the threshold.

Curriculum, direct-training, transfer, and ablation comparisons use the same
locked evaluation suite and selection rule. Training returns are not substituted
for canonical held-out evaluation outcomes.

## Runtime and resource protocol

Runtime results name the device, driver/runtime and library versions, numeric
precision, batch shapes, environment count, horizon, included policy work,
recording profile, and repetition count.

For the current foundations work, **GPU speed and efficiency are the performance
qualification target**. CPU checks establish correctness and compatibility; they
do not impose a separate CPU speed target. Preserve historical CPU measurements
with their workload and date. Host work still counts when it is part of the
complete GPU workflow, including setup, transfers, output, and rating analysis.

Report separately:

- setup and first-call time, separating compilation where the measurement can;
- warmed steady-state environment throughput;
- policy inference throughput/latency;
- optional communication-model time and cost;
- host capture/event/metric overhead when enabled; and
- peak device and host memory where relevant; and
- host/device transfers and output bytes where those costs are affected.

Canonical SharedObs composition, projection, and compiled actor-front-end cost
are included in the declared runtime boundary. A separately labelled custom
diagnostic may measure NoSharedObs compatibility, but it is not an official
benchmark stratum or evidence for a separate runner or evaluation stack.

Fix warm-up and timed repetition counts in the manifest. Synchronize GPU work
before reading elapsed time. Compare equal inputs, policy work, retained outputs,
and source/asset identities; otherwise a faster run may simply do less work.
Check compilation reuse when ordinary values change without changing shapes or
dtypes. Do not call first-call time pure compilation without a way to separate it.
Label capture costs explicitly when they are included in a training-throughput
number. A disabled evaluation path performs no
host transfer, model validation, event decoding, logging, or trajectory
retention. In CP3, disabled means that no observer or evaluation context is
constructed and the caller never invokes capture. An explicitly constructed
`training_light` or `debug` observer is enabled work: it streams the public
semantic view without retaining trajectory history. The
`evaluation_metric_complete` and `scenario_metric_complete` profiles retain
the exact in-memory `T + 1` frame / `T` transition prefix; scenario capture
continues to require scenario identity. Debug adds no private-state payload.

## Artifact, replay, and reporting requirements

The normative version-1 replay normal form and its non-circular artifact graph
are specified in [Standard Evaluation Replay Format](replay_format.md). Replay
stores the context once plus exact `T + 1` frames and `T` transitions,
completion, processing status, and a content-addressed metric-report reference.
The context's schema map remains the exact eight CP2 bindings; replay-envelope
bindings live in the replay header. Structural model validation alone is not a
semantic replay-validity claim: every loaded artifact must pass the explicit
whole-artifact validator over frame zero and all adjacent four-record units.

Canonical replay persistence is local, bounded, and fail-closed. The loader
accepts only canonical finite UTF-8 JSON in regular nonsymlink files, rejects
duplicate keys, excessive depth/size, unknown versions, and digest or semantic
mismatches, and performs no JAX/backend, simulator, policy, capture, or device
work. Bundle publication writes the content-addressed metric report first and
the replay last through same-directory durable no-clobber publication. A replay
is still renderable when a report sidecar is absent, but a present invalid or
foreign sidecar is an error and metric completeness is never inferred.
The V1 filesystem backend requires descriptor-relative POSIX no-follow
operations and fails closed on platforms that cannot keep every path component
and both bundle publications bound to one opened directory inode.

Scenario and actor-POV companions use the same finite canonical JSON,
descriptor-bound nonsymlink path walk, size/depth limits, and atomic no-clobber
publication. A POV save must validate its completed replay reference. A
historical V1/V2 scenario save or load validates both its replay and metric-report
evidence joins. Historical V3 scenario records join replay V2 directly. Current
V4 scenario records join replay V3. Neither family has a metric-report field. A structurally valid but foreign record is not accepted as
a local scenario result.

Canonical V2 scenario loading and saving remain JAX-free and establish artifact
and evidence validity, not official product acceptance. An official consumer
must additionally invoke `validate_official_scenario_evaluation_record_v2`.
That separately named host gate rehydrates the exact carried configuration and
initial state, applies the current core product and curated-state validators,
requires the canonical SharedObs mode and projection, and checks the exact
configured-roster availability topology on every replay frame.
Historical V3 readers preserve this separation:
`validate_official_scenario_evaluation_record_v3` applies the same product,
initial-state and all-frame SharedObs checks to replay V2. Current V4 scenario
records join replay V3 and use the corresponding V4 official gate. No mutable draft, filename, or successful transport round trip can
substitute for the explicit official gate.

Rollout completion, evaluation-processing validity, and per-statistic endpoint
observation remain independent in both live and replay-loaded analysis. A
processing failure never rewrites a provably complete rollout, and an
infrastructure prefix never becomes scientific right censoring.

Official results retain enough information to reproduce every reduction:

- resolved configuration and static-mechanics catalog once per episode;
- catalog-digested `global_recipient_slot_by_actor_and_target_action`,
  `global_slot_by_actor_and_ally_observation_row`,
  `global_slot_by_actor_and_enemy_observation_row`, and
  `unit_direction_vector_by_movement_action` mappings, with aligned action and
  relation-axis vocabularies;
- schema, code, suite, manifest, information-regime, projection, critic,
  reward, and shaping identities;
- `T + 1` semantic frames and `T` transitions when replay retention is enabled;
- raw metric sufficient components and complete cell/subject keys;
- rollout completion, observer-processing failure, per-statistic endpoint
  observation, result eligibility, and artifact-validation status; and
- deterministic links from aggregate rows to source artifacts.

The catalog mappings, joined through the episode roster, are the sole artifact
authority for translating actor-relative action/mask columns and
relation-local observation rows into stable agent identity. Offline consumers
must not import private simulator lookups or recreate indexing formulas. The
ally/enemy row mappings cover unit features, visibility, visible action
history, and own-team/opponent-team local-slot axes in spawn-lifecycle
observations; the aligned vocabulary names the two lifecycle team-axis
entries.

An independently shareable NoSharedObs POV artifact copies only the selected
recipient's rows and the minimal recipient-local forms of those mappings. It
uses separate submitted-int32 and accepted-category action records so rejected
out-of-domain intent is preserved rather than normalized. Its local cues may
describe only changes present in authorized adjacent rows plus own action,
reward, mask, and done truth. Full replay provenance remains in a separate outer
reference; privacy equality applies to recipient-content bytes, not to that
truthful provenance wrapper.

Evaluation frames store base observations and masks once. They do not duplicate
materialized SharedObs. SharedObs actor inputs remain reproducible from the
same-epoch base sensor projections, source-axis/provenance mapping, required
recipient-by-source availability inputs, and recorded actor-input projection
version. Historical source banks use their recorded source/global-slot mappings
rather than current code constants. Current banks use five own-team sources and
ten relative candidates: five allies, then five enemies. Their feature shape is
`(5, 10, 58)`, visibility shape is `(5, 10)`, objective shape is `(5, 8, 12)`,
and source-availability shape is `(5,)`. The selected actor's own observation
arrives separately; its own source row is unavailable. Current replay capture derives availability
from the captured active mask and team IDs through the same canonical topology
helper used by policy input construction. Saved availability matrices remain the
authority when reading historical artifacts. Learned encoder tensors remain a
separate Milestone 10 contract.
World-state critic inputs and privileged evaluation snapshots are separate
contracts and never leak into actor inputs.

For official evidence, the episode mode and projection must equal the canonical
SharedObs values defined above, and every frame's recorded matrix must equal
the configured-active, same-team, off-diagonal matrix derived from the frozen
roster. This official all-frame equality check is stricter than generic replay
validation, which continues to admit dual-mode compatibility and a
permitted SharedObs subset topology for custom work.

The canonical `execution_information_mode` values are `shared_obs` and
`no_shared_obs`; the historical design PDF's earlier disabled-sharing
terminology is explicitly superseded without modifying that historical
artifact. The frame schema has an optional, explicit Boolean availability
matrix with axes
`(recipient_global_slot, sensor_source_global_slot)` and exact shape `(10,
10)`. Conditional validation requires the matrix for `shared_obs` and forbids
it for `no_shared_obs`. Diagonal, cross-team, inactive-recipient, and
inactive-source entries are false. Neither regime stores a materialized
SharedObs actor-input projection in the evaluation frame.

The recorded availability matrix is routing metadata. A policy receives only
its five own-team source entries, with every unavailable source cleared. Global
slots, hard team IDs, roster ownership and catalog mappings stay outside the
current policy input. Feature column 3 means `is_enemy`: self and allies are
zero, visible enemies are one, and unseen rows remain zero. Existing visibility
and activity masks distinguish hidden enemies from allies and unused padding.
`self_ally_index` is a scalar own-team row number from 0 to 4; inactive actors
receive zero. Shared networks normally condition on `self_features`.

The current contract uses base observation/frame V2, episode context V3,
SharedObs projection V2, NoSharedObs projection V3 and replay V3. Historical
records keep their original team-number feature and projection meaning. A run
cannot resume with a different input contract. See
[A37](../design/specification_amendments.md#a37-relative-policy-identity-and-versioned-recordings)
and the [policy input guide](workflows.md#policy-inputs).

Execution mode and compatible projection remain episode-wide context
authorities, not per-assignment fields. All configured active policy assignments
must therefore be homogeneous. The historical V1 context, replay and metric
report families remain immutable. Their version numbers do not authorize mixed
execution.

Milestone 6 evaluation records use a single normalized authority for submitted
and accepted actions inside `TransitionFactsV1.action_acceptance_facts`; the
transition record does not duplicate them. The normalized model preserves
every core transition-fact subtree and leaf name exactly. Each transition
stores `canonical_reward_by_agent` with ten entries and may store
`canonical_reward_by_team` with two entries. Reward-shaping values remain
trainer-owned and excluded even though the immutable context records the
shaping configuration identity.

Static slot truth belongs to the episode context: fixed-slot identity/topology
is in the roster, while body radius, movement, interaction ranges, maximum
health, and recovery mechanics are in `resolved_env_config.slot_mechanics`.
The global analysis snapshot contains dynamic state only. The context contains
exactly ten discriminated policy-assignment rows, and every inactive slot uses
the explicit `not_applicable` variant. Durable policy and code identities never
rely on local paths. Every revision requires `source_tree_digest`;
`dirty_patch_digest` is additionally required exactly when `is_dirty` is true.
Catalog digests use finite-only canonical JSON with ASCII identifiers,
recursive `-0.0` normalization, sorted keys, compact separators, and UTF-8
encoding.

Cross-record validity is established by one public four-record validator over
the context, transition-start frame, transition, and successor frame. It
checks adjacency, identity, lossless core-recipient `-1`/JSON `null`
normalization with `has_recipient` agreement, padding, catalog joins, and exact
equality with a newly decoded canonical event tuple. The discriminated V1 event
union has exactly 24 atomic variants. At rank 50,
`AgentLeftCombatEventV1` follows countdown reset and precedes regeneration for
the canonical alive-recipient countdown edge from one to zero. At rank 90,
`AgentDiedEventV1` records the newly dead recipient before one
`LethalDamageContributionEventV1` per ordered
authoritative positive source; contributor records are not killer, last-hit,
or complete historical credit. Aura attachments identify direct
transition-start covering emitters rather than asserting causal credit.
Rank 120 uses family-specific coordinates: team waves sort by `(120,
team_index, -1, wave_subtype, neutral_source)` and realized respawns by `(120,
configured_team_index, agent_global_slot, respawn_subtype, neutral_source)`.
Therefore each team's wave precedes its realized agents and team groups cannot
interleave.
At rank 130, `TeamDeathmatchScoreChangedEventV1` records one positive
authoritative score edge with previous and successor values. At rank 140,
`TeamDeathmatchCompletedEventV1` records the authoritative win/draw/loss and
completion basis. Live and replay consumers preserve both event payloads
without reconstructing score or outcome.
Pydantic serialization/revalidation alone proves structural roundtrip, not
semantic trajectory validity.

Dashboard, CSV, JSON, and paper-table layers reference metric IDs and protocol
versions. They do not duplicate or silently modify formulas. Presentation
rounding occurs only after aggregation.

Illustrative replay selection is also protocol-owned. A suite or manifest
predeclares deterministic categories such as median seed/cell performance,
upper/lower-tail examples, scenario successes/failures, and infrastructure or
policy failures; it retains the complete candidate index and tie-break rule.
Qualitative replay examples never replace aggregate results and are not chosen
ad hoc to support a preferred narrative.

## Required validation and release audit

Before a suite is used for an official claim, verify:

1. every result maps to a recognized metric definition and frozen suite;
2. all scheduled cells and weights are accounted for;
3. train/validation/locked-test boundaries and checkpoint selection are clean;
4. zero opportunities, structural inapplicability, attribution ambiguity,
   invalid artifacts, and insufficient data remain distinct;
5. team/side swaps and common-condition pairing behave as declared;
6. hand traces cover simultaneous effects, duplicate classes, ties, partial
   runs, censoring, and failures;
7. replay/model roundtrips preserve sufficient components;
8. aggregation reproduces from raw rows without simulator-rule reconstruction;
9. adversarial denominator and reward-gaming cases have visible companions;
10. canonical SharedObs mode, projection, and all-frame availability are exact;
    privileged sources cannot leak, and NoSharedObs evidence cannot enter an
    official aggregate;
11. the four North Stars receive an explicit `PASS`, `APPROVED TRADEOFF`, or
    `BLOCKED` verdict;
12. official/custom ownership and canonical-condition status remain separate
    and appear truthfully in every report;
13. each accepted scenario exactly joins independently approved,
    content-addressed layout and authored-initial-condition identities,
    resolved configuration, explicit roster, configured-active slots,
    fixed-slot role template, stable matched seed schedule, one unique realized
    schedule coordinate, horizon, initial state, and endpoints; any DevClient
    source identifies one exact saved revision, loaded context passes core
    structural/product parity, current browser buffers and moving revision
    aliases are rejected, and policy binding cannot replace those facts; and
14. the permissive standard-layout factory accepts every structurally valid
    roster while the canonical factory resolves exactly the frozen mirrored
    five-class 5v5 evaluation configuration;
15. each controlled contrast binds the same scenario, pressure-controller
    version, SharedObs contract, seeds, sides, budget, selection rule, and
    endpoint to a full method and its one declared ablation, with independently
    trained paired runs as the replication unit;
16. the complete public scenario closure is digest-disjoint from training and
    validation manifests and has not influenced any adaptive decision;
17. a frozen Big 12 snapshot contains exactly twelve fixed systems and one
    Elo value per method; a canonical run covers those twelve alone or adds
    one challenger, with 66 or 78 matchups respectively, one approved uniform
    budget, complete paired coverage, stable origin identities, and the full
    raw outcome/failure matrix; any research budget override is labelled and
    cannot qualify promotion;
18. rows 1–11 retain three independent training runs and select one tournament
    system using only the predeclared validation rule; Qwen-Five has separately
    passed its measured cost and compatibility gate;
19. the rating estimator, uncertainty, convergence/failure behavior, verified
    reuse, monthly admission/publication, and immutable snapshot identities
    have passed their explicit gates; all remaining official launch decisions
    named above are approved and qualified; and
20. every admitted system is reproducible from immutable implementation,
    configuration, checkpoint, population, and selection provenance, and no
    training consumer resolves a mutable `latest_big_12` alias.

Partition status in this audit comes from the owning distribution, suite,
scenario, or experiment manifest. It must never be inferred from a map asset's
filename, directory, mutable draft identity, or authoring history.

Any unresolved Tier-1 semantic, leakage, attribution, or pseudoreplication
finding blocks an official benchmark claim.

## Methodological anchors

The protocol follows the cooperative-MARL standardization recommendations in
[Gorsane et al.](https://arxiv.org/abs/2209.10485), the robust aggregate and
uncertainty guidance in
[Agarwal et al.](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html),
and the held-out partner/opponent principles illustrated by
[ZSC-Eval](https://arxiv.org/abs/2310.05208) and
[Melting Pot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d3ea22480873b389a3365d711eb1e91-Abstract-Datasets_and_Benchmarks_Track.html).

These references inform the protocol; MARL-BattleGrounds' exact metric
semantics and task contracts remain defined by this repository.
