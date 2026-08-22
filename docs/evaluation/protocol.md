# MARL-BattleGrounds Evaluation Protocol

## Status and authority

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** This is the controlling public
> evaluation protocol.

This document is the normative protocol for evaluating MARL-BattleGrounds
policies. It owns evaluation populations, cells, frozen
weights, aggregation,
uncertainty, checkpoint selection, information-regime separation, cross-play,
controlled scenarios, runtime measurement, and failure/censoring treatment.
The companion [metric specification](metric_specification.md) owns stable
metric meanings, sufficient components, eligibility, attribution, and allowed
interpretations. Accepted departures from the original design PDF are recorded
in the [specification amendments](../design/specification_amendments.md).

The protocol does not prescribe one universal experiment. It prescribes the
information every evaluation suite and experiment manifest must freeze so that
a reported result is reproducible and scientifically interpretable.

## Protocol objects

The conceptual public contract has four layers. These names do not require a
production registry during Milestone 6.

### Metric definition

A versioned metric definition is invariant across studies. It identifies the
question, subject, data authority, units and amount stage, opportunity,
sufficient components, reduction, zero-opportunity behavior, attribution,
allowed claim, gameability companions, and validation state.

Changing a formula, denominator, eligibility rule, amount stage, attribution,
or semantic scope creates a new metric version. Changing an opponent pool,
cell weight, confidence interval, or checkpoint does not.

### Evaluation suite

An `EvaluationSuiteV1`-equivalent declaration freezes:

- suite ID, version, canonical digest, ownership classification as `official`
  or `custom`, separate canonical-condition status, and intended scientific
  claim;
- battleground identity and version plus the resolved episode-configuration
  contract;
- execution-information assignment profile, including each configured active
  assignment's regime and actor-input projection version; in V1 this profile
  is represented by one homogeneous episode-wide pair;
- critic-information regime, canonical reward mode, and shaping configuration;
- primary, secondary, exploratory, and diagnostic metric IDs;
- promoted content-addressed layouts/maps and authored initial conditions,
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

An experiment manifest freezes study-specific choices before locked-test
evaluation:

- algorithms, policy architectures, training configurations, and code/artifact
  revisions;
- each learned checkpoint's fixed execution-information regime, actor-input
  projection version, and compatible compiled actor-front-end contract;
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

A `MetricResultV1`-equivalent row carries:

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

Milestone 6 CP3 realizes the generic host-side portion of this contract without
activating a universal metric registry. An opt-in `EvaluationEpisodeObserverV1`
passes each reducer one already validated coherent view consisting of the
episode context, transition-start frame, transition, and successor frame.
Reducers are deterministic, immutable, copy-on-write consumers; they never
receive unresolved frame references or permission to reconstruct simulator
rules. `EvaluationMetricReportV1` stores its episode context once and joins raw
statistic rows back to that context rather than repeating policy, seed, task,
scenario, information-regime, reward, and shaping provenance on every row.

The observer distinguishes validated transition count from successfully
processed transition count. A failed append or reducer operation poisons the
observer, preserves the last validated gap-free prefix, and cannot publish a
partially updated statistic set. `EvaluationProcessingStatusV1` reports that
failure independently of `EvaluationEpisodeCompletionV1`; a processing failure
after a completed rollout never rewrites the rollout as failed. There is no
logging sink, file writer, replay envelope, runner framework, or core callback
in CP3.

## Episode, training, evaluation, and scenario ownership

An episode configuration resolves the fixed simulator inputs for one rollout.
It is selected by a training distribution or fixed by an evaluation suite or
scenario. It is not itself a training distribution, evaluation population, or
scenario. [Amendment A15](../design/specification_amendments.md#a15-episode-configuration-and-experiment-distribution-ownership)
defines structural validity and the permissive Team Deathmatch roster
contract.

Milestone 11 owns all training selection. Default direct Team Deathmatch
training uses the canonical mirrored five-class 5v5 roster and the same task
mechanics, lifecycle rules, score threshold `K`, horizon `H`, and canonical
reward as official evaluation, but it samples from the separately approved
training-map distribution and training seed schedule. Researchers may replace
that default with any distribution over structurally valid episode
configurations. An optional curriculum is a checkpointable, stateful selector
over the same contract; its benchmark 1v1–5v5 path uses explicitly approved,
handpicked rosters rather than an exhaustive composition grid. Map identities
selected into the training distribution and the exact curriculum roster, map,
weight, retention, opponent, and transition choices remain future Milestone 11
decisions.

An evaluation suite owns a frozen population of resolved configurations and
assignments. A scenario owns its resolved episode configuration, explicit
fixed-slot roster, initial state, role template, horizon, matched seed schedule,
and quantitative endpoints. The schedule is a stable multi-attempt definition;
each episode's realized seed record joins to exactly one declared schedule
coordinate. Binding policy A to Team A and policy B to Team B, or binding
policies per slot, fills the scenario's frozen roles; it cannot replace the
roster or any other scenario-owned condition.

Development scene-authoring drafts are mutable design inputs, not evaluation
artifacts. A candidate becomes eligible for selection only after host-side
normalization, product/scenario validation, content addressing, and an
owner-approved promotion that assigns its permanent identity and version.
Official evaluation and scenario records reference those promoted
content-addressed layout and authored-initial-condition identities and join
them independently to the resolved configuration and realized initial state. A
filename, browser export, mutable draft ID, or successful preview is not an
admissible identity or validity proof.

The current `ResolvedScenarioSpecificationV1` does not yet bind an explicit
fixed-slot roster, exact role template, or matched schedule identity and
membership rule. Its resolved-config digest contains no roster rows, and
`EvaluationSeedProtocolV1` represents one realized episode rather than the
schedule. Before an M7 scenario can be accepted as official, C2 must version
that contract, validate exact roster/role joins to
`EvaluationEpisodeContextV1`, prove that each realized seed record occupies
exactly one declared schedule coordinate, and enforce the core structural and
product config invariants on loaded context. The versioned specification must
also freeze and join the promoted content-addressed layout and authored initial
condition independently from the resolved-config digest. It must reject any
policy binding that changes the layout, initial condition, resolved
configuration, roster, configured-active slots, role template, schedule, or
realized coordinate. Scenario identity plus a subset check over role names is
insufficient.

No training distribution is inferred from an evaluation suite or scenario.
Locked evaluation and scenario maps, seeds, opponents, initial states, and
other held-out material are ineligible for policy or curriculum updates.

## Common policy and evaluation lifecycle

[Amendment A12](../design/specification_amendments.md#a12-common-milestone-1012-policy-pipeline-spine)
requires Milestones 10–12 to extend one common pipeline spine. Episode
specifications selected by direct/custom training distributions or curricula,
and episode specifications fixed by evaluation suites or scenarios, enter the
same versioned policy-assignment and seed protocol. Both execution-information
regimes use the same reset, base observation, exact mask, legal-action
realization, joint-action assembly, core-step, transition, completion,
capture/replay, and metric semantics. Evaluation/scenario episodes use M10's
host adapter; M11-selected training episodes use M12's JAX
rollout/update/checkpoint adapter. Training selection does not create a runner
or trainer, and training does not route through the evaluation/scenario host
adapter.

Only authorized-input composition, learner-input projection, the compatible
compiled actor front end, checkpoint compatibility validation, measured cost,
and manifest/report stratum may differ by execution-information regime. A
regime must not select a separate runner, trainer, evaluator, scenario
lifecycle, RNG protocol, action realization, capture path, replay path, or
metric implementation. Milestone 10 owns the versioned episode and
policy-assignment contracts, semantic-asset promotion adapters, seed-schedule
schema and named derivation protocol, evaluation/scenario host runner,
capture/replay, and failure-semantics integration. Milestone 11 selects
promoted partition-neutral maps into direct, custom-distribution, or
curriculum training populations and feeds their resolved specifications into
the common training spine. Milestone 12 consumes those selections through the
common JAX rollout/update path and owns the regime-specific compositor, learner
projections, compiled actor front ends, and trainer/checkpoint integration; it
does not own a roster catalog, training-selection policy, promotion boundary,
or mutable development-draft loader. M12 rejects held-out evaluation/scenario
assets before training execution.

Every learned checkpoint is fixed to one `execution_information_mode`, one
actor-input projection version, and one compatible compiled actor-front-end
contract. Compatibility validation rejects a mismatch before compilation,
device allocation, or execution. An explicitly declared cross-regime
initialization is a transfer or out-of-distribution intervention, not a mode
switch, and receives separate provenance and reporting.

The current `EvaluationEpisodeContextV1` contract is immutable and
homogeneous: its single episode-wide `execution_information_mode` and
`actor_projection` apply to every configured active policy assignment. V1
mixed execution—any episode assigning SharedObs to some active slots and
NoSharedObs to others—is scientifically ineligible even if a runtime can
mechanically assemble its actions. It must not enter an official evaluation
cell, controlled-scenario claim, replay publication, metric report, or
cross-play aggregate.

Mixed execution remains ineligible until Milestone 10 defines a V2 context and
artifact family with per-active-slot execution-mode and actor-projection
provenance plus sufficient per-recipient availability authority. V2 must feed
the same policy/scenario/capture/replay/metric lifecycle; it must not mutate or
reinterpret V1. Its homogeneous/mixed episode profile is derived from the
configured-active per-slot assignments, never edited independently. An
explicit V1-to-V2 migration may replicate V1's episode-wide mode and projection
across configured-active slots and mark inactive slots not applicable, so it
can create only a homogeneous V2 profile; it fails when required versioned
provenance cannot be established and never mutates V1 bytes or digests. Once
V2 exists, focal SharedObs versus opponent NoSharedObs and focal NoSharedObs
versus opponent SharedObs are distinct directional cells, each with
task-appropriate side swaps. They are never pooled with each other or with
either homogeneous regime.

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
     x execution_information_assignment_profile
```

Use a declared not-applicable sentinel when the evaluated setting has no
separable cooperative partner or no adversarial opponent. Never collapse these
axes into one “other policy” coordinate: varying a partner while also varying
the opponent confounds cooperative generalization with adversarial robustness.

Resolved episode configuration, scenario version, policy IDs,
critic-information regime,
reward mode, shaping configuration, code revision, static-catalog digest, and
actor-input projection version remain mandatory provenance. They become
separate cell axes whenever varying one can change the intended estimand.

Battleground identity/version and the execution-information assignment profile
are separate reporting strata by default. V1 maps its one episode-wide
mode/projection pair to the corresponding homogeneous profile. SharedObs and
NoSharedObs results are never pooled. Raw task-score units are never pooled
across battlegrounds. A cross-battleground scalar requires a separately
versioned normalization and suite contract.

For V1, the cell's `execution_information_mode` is the one homogeneous mode of
every configured active assignment. A future eligible V2 mixed-regime cell
must also retain the full directional per-assignment mode/projection pattern as
cell provenance; naming only the focal policy's regime is insufficient. Its
homogeneous/mixed profile is derived from that pattern rather than supplied as
an independent cell authority.

## Suite construction and frozen weights

Each suite declares a finite set of cells and nonnegative target weights
`w_c` summing to one within each reported task and execution-information
assignment-profile stratum. Equal cell weights are the default. A
task-declared target distribution is allowed when it is motivated and frozen
before evaluation.

Weights express the target population, not the number of episodes that happened
to finish. A failed cell, zero-opportunity cell, or missing artifact does not
cause the remaining weights to be silently renormalized. The result instead
uses the suite's declared failure/missingness rule and exposes coverage.

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

Do not average cell rates. Do not pool opportunities across training runs. Do
not renormalize weights around cells with zero realized opportunities. Preserve
each `n_bar`, `d_bar`, and defined cell rate for audit.

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

Uptime-like quantities preserve qualifying and eligible agent-steps. Event and
episode distributions remain long-form, with cell/run identities attached.
Default summaries are the median and interquartile range; tail quantiles are
used when scientifically motivated. Min/max is not a default robustness or
quality statistic.

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

Independent training runs—not an arbitrary universal number—determine
inferential strength. Each manifest declares a seed budget and a precision or
stopping rule before evaluation. When too few independent runs support a
defensible uncertainty estimate, label the result descriptive; do not disguise
episode replication as additional seeds.

For multi-task or multi-suite comparisons, report task-level results first.
Interquartile mean, optimality-gap, and performance-profile summaries with
stratified bootstrap uncertainty are appropriate only when normalization and
task sampling are declared. They do not replace the underlying task results.

Only blocks labeled `primary_confirmatory` define the default confirmatory
family. A descriptive coordination block does not become confirmatory merely
because it appears on the compact card. Key secondary endpoints form named
families by scientific claim. Exploratory, diagnostic, scenario, and post-hoc
slices are visibly labeled and cannot be promoted after looking at locked-test
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
the map document. Mutable development drafts are ineligible for every such
population; manifests may reference only owner-promoted, versioned,
content-addressed assets or their resolved contract representations.

An official evaluation suite or scenario is never a default training
distribution. Assets designated locked for evaluation or scenario evidence
must not be sampled for policy updates, curriculum transitions, opponent
adaptation, reward tuning, or checkpoint selection.

- Training data may update policies and curricula.
- Development data may debug implementations and metric code.
- Validation data may select checkpoints, tune hyperparameters, and validate
  predeclared scenario thresholds or other analysis parameters.
- Locked-test data may evaluate a frozen decision only. It may not choose the
  metric, formula, threshold, opponent pool, checkpoint, reward shaping, or
  presentation cutoff.

The checkpoint-selection rule must be executable from validation information
alone and must state tie-breaking. Best-of-many test selection is prohibited.
When a locked-test defect requires rerunning, preserve the failed attempt,
explain the defect, and revise the manifest rather than overwriting history.

## Completion, failure, missingness, and censoring

Artifact validity, observer processing, rollout completion, and per-statistic
endpoint observation are separate dimensions. None may be inferred from or
collapsed into another.

### Artifact validity

An artifact is valid only when schemas and catalog versions are recognized,
digests match, identities and adjacent indices are gap-free, payloads are
finite and shape-correct, catalog axis mappings are complete/aligned and join
every fixed global slot through the episode roster, and completion metadata is
internally consistent. Invalid artifacts contribute to failure/QC reporting
but no tactical metric.

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
declared reducer and the final report was validated atomically. Processing
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

Censoring means a valid endpoint was not observed within a declared scientific
window; it is not an episode completion state or a synonym for corrupted data.
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

Missingness policies state whether a claim targets all scheduled runs,
successfully trained runs, or valid completed evaluations. The distinction is
part of the estimand and cannot be chosen after seeing failures.

## Symmetry, sides, and common conditions

Both teams use the same metric definitions. Suites use side swaps and paired
layouts/seeds when the task permits them. Side effects are reported and remain
part of the cell coordinates. The two sides of one match are paired views of a
single stochastic event.

For self-play or policy-versus-policy evaluation, policy identity and side are
not conflated. A symmetric matrix records both assignment directions or uses a
task-justified symmetry reduction that remains recoverable.

After the V2 mixed-regime contract exists, execution-information assignment
direction is likewise not conflated with policy identity or side. Both
SharedObs-versus-NoSharedObs directions remain recoverable and receive the same
side-swap discipline. V1 remains homogeneous and cannot carry these cells.

## Compact scorecard reporting

The [metric specification](metric_specification.md#presentation-budgets)
exclusively owns the contents and budgets of the primary team and agent/class
cards. An evaluation suite instantiates that scorecard for each task and
execution-information assignment profile; V1 uses the corresponding
homogeneous profile. This protocol does not redefine the scorecard's metric
blocks. The suite and manifest freeze which eligible candidate fills each
optional slot and place those primary blocks in the confirmatory endpoint
family before locked-test evaluation.

Every primary table footer reports suite/manifest versions, independent
training runs, cells, episode schedule, defined/undefined counts, failure and
truncation rates, cell weighting, and uncertainty method. Raw sufficient
components remain exportable.

## Controlled-scenario protocol

Context-sensitive tactical claims use controlled quantitative scenarios rather
than episode-wide proxies. Each versioned scenario freezes:

- the behavioral hypothesis and eligible policy roles;
- promoted content-addressed layout and authored-initial-condition identities;
- its resolved episode configuration and exact fixed-slot roster;
- initial state, role template, opponent or scripted pressure, horizon, and
  matched seed schedule;
- one primary quantitative endpoint;
- at most two secondary margins;
- explicit safety, role, or behavior violations;
- success, failure, terminal, and censoring semantics; and
- replay retention and blinded review sampling.

Policy assignment is an execution binding, not scenario construction. A
Team A/Team B convenience binding and a lower-level per-slot binding must both
respect the scenario's fixed roster, configured-active slots, and role
template. Changing those facts creates a different versioned scenario rather
than an override of the current one.

Scenario validation must also join the explicit fixed-slot roster and role
template carried by the resolved scenario definition to the episode context,
then prove that the episode's realized seed record is exactly one member of the
scenario's stable matched schedule. A set/subset comparison of role labels or
a digest of one realized seed record does not establish those identities.

“Single-shot” prohibits learning or adaptation across attempts. It does not
reduce a stochastic evaluation to one episode. Policies are compared over the
same frozen scenario seeds when possible.

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

Cross-play applies a frozen population protocol to a selected base task
endpoint. It does not duplicate every tactical metric. Cooperative partners and
adversarial opponents are separate experimental axes.

The V1 cross-play tensor is homogeneous in its episode-wide
`execution_information_mode`. A later V2 tensor may add a frozen directional
per-assignment regime pattern, but only after V2 supplies truthful
per-active-slot provenance; the two mixed directions are separate tensor cells
and are not pooled.

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

## Learning and sample efficiency

Training curves are evaluated through frozen periodic held-out evaluations.
Ordinary curriculum rollouts do not run the full evaluation suite.

Default direct training, arbitrary researcher-defined training distributions,
and curriculum training share one rollout, batch, optimizer, update, and
checkpoint lifecycle across information regimes. The curriculum differs only
by retaining selector state and emitting the next ordinary episode
specification. Regime-specific input composition, learner projection, and
compiled actor front ends plug into that lifecycle; neither roster selection
nor information regime creates a separate trainer. A learned checkpoint is
emitted for exactly one regime/projection/front-end contract, and the learner
interface must accept every structurally valid fixed-slot roster without an
input-layer or simulator-schema change.

Every learning result records:

- environment transitions;
- active-agent decision transitions;
- optimizer updates and samples consumed when applicable;
- wall-clock and compute/hardware provenance;
- evaluation checkpoint schedule; and
- task, roster size, information regime, reward mode, and shaping identity.

Both environment and active-agent decision transitions are required because a
1v1 and a 5v5 environment transition expose different amounts of agent
experience. Every area-under-curve or time-to-threshold result names its x-axis,
fixed horizon, interpolation convention, threshold, and treatment of runs that
never reach the threshold.

Curriculum, direct-training, transfer, and ablation comparisons use the same
locked evaluation suite and selection rule. Training returns are not substituted
for canonical held-out evaluation outcomes.

## Runtime and resource protocol

Runtime results declare device model, driver/runtime and library versions,
precision, batch/vectorization shape, environment count, horizon, policy
inclusion, capture profile, and repetition count.

Report separately:

- first compile time;
- warmed steady-state environment throughput;
- policy inference throughput/latency;
- optional communication-model time and cost;
- host capture/event/metric overhead when enabled; and
- peak device and host memory where relevant.

Any regime-specific composition, projection, or compiled actor-front-end cost
is reported as a measured stratum of this same runtime protocol, not as
evidence for a separate runner or evaluation stack.

Warm-up iterations and timed repetitions are fixed in the manifest. Evaluation
capture must not be included in a claimed ordinary-training throughput number
unless the label explicitly says so. A disabled evaluation path performs no
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
scenario save or load must validate both its replay and metric-report evidence
joins; a structurally valid but foreign record is not accepted as a local
scenario result.

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
version.
World-state critic inputs and privileged evaluation snapshots are separate
contracts and never leak into actor inputs.

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

This V1 mode and its compatible actor projection are episode-wide context
authorities, not per-assignment fields. All configured active policy
assignments must therefore be homogeneous. The current V1 context, replay, and
metric-report family is immutable and rejects mixed-regime scientific claims;
adding per-active-slot mode/projection provenance requires the Milestone 10 V2
family described above.

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
union has exactly 23 atomic variants. At rank 90, `AgentDiedEventV1` records the
newly dead recipient before one `LethalDamageContributionEventV1` per ordered
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
10. information regimes and privileged sources cannot leak or pool;
11. the four North Stars receive an explicit `PASS`, `APPROVED TRADEOFF`, or
    `BLOCKED` verdict;
12. official/custom ownership and canonical-condition status remain separate
    and appear truthfully in every report;
13. each accepted scenario exactly joins promoted content-addressed layout and
    authored-initial-condition identities, resolved configuration, explicit
    roster, configured-active slots, fixed-slot role template, stable matched
    seed schedule, one unique realized schedule coordinate, horizon, initial
    state, and endpoints; loaded context passes core structural/product parity,
    mutable drafts and unpromoted candidates are rejected, and policy binding
    cannot replace those facts; and
14. the permissive standard-layout factory accepts every structurally valid
    roster while the canonical factory resolves exactly the frozen mirrored
    five-class 5v5 evaluation configuration.

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
