# MARL-BattleGrounds Specification Amendments

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** This document records the
> accepted amendments to the historical design PDF.

This document changes `MARL_BGs_Design_Document.pdf`. The PDF remains the
unaltered historical architectural blueprint; this file is the controlling
public source where the two disagree. An amendment changes only the clauses it
names. Unmentioned PDF requirements remain in force.

The amendments below were drafted before Milestone 6 Step 5 implementation.
They deliberately favor the four project North Stars: researcher-centricity,
low sample complexity, meaningful tactical and strategic team behavior, and
professional MARL/software engineering.

## A1. Evaluation information regimes

**Classification:** risky, accepted design drift.
**Supersedes:** R20; Sections 2.4, 2.4.1, 2.4.8, 2.4.11, 2.6.18, 2.8.5,
2.13.8, 2.15.7, 2.16.2, and 2.16.19; Appendix A.11; and baseline or paper
clauses that use the historical disabled-sharing label, make that regime the
default, or make SharedObs merely optional.

SharedObs is the approved future default execution-time actor-information
regime. It is not current runtime behavior and does not become active until its
versioned learner-input projection is implemented, performance-tested, and
accepted before canonical baseline training.

NoSharedObs remains an official first-class regime. It must remain selectable
and must be evaluated and reported separately from SharedObs. Results from the
two regimes must never be pooled into one benchmark cell or summary.

The editable specification uses **NoSharedObs** and `no_shared_obs`. The
historical PDF's earlier disabled-sharing wording is superseded rather than
rewritten in the PDF itself. NoSharedObs means only that the SharedObs
compositor is disabled; it does not claim that an algorithm has no
communication mechanism of its own.

The canonical implementation boundary is:

- `execution_information_mode` is the extensible learner-side setting whose
  current values are `shared_obs` and `no_shared_obs`;
- a command-line Boolean may be a convenience alias, but is not the canonical
  persisted representation;
- simulator state, dynamics, rewards, action semantics, base observations, and
  masks do not change with this selection;
- in `shared_obs`, each already-authored same-decision-epoch base sensor
  projection is factored once into a fixed-shape, roster-joined sensor-source
  bank that preserves stable source identity and each source's normal
  visibility, line of sight, padding, and lifecycle redaction;
- each actor's shared input applies a versioned Boolean
  recipient-by-source information-availability mask to that bank, so teammates
  may receive different source subsets rather than one identical team tensor;
- the compositor never recomputes geometry, visibility, line of sight, or
  redaction, and the actor's own base observation remains separate;
- in `no_shared_obs`, the learner returns the actor's base observation and
  bypasses material shared-bank and recipient-by-source-mask construction;
- teammate masks, previous-action/history fields, recurrent state, policy
  memory, rewards, transition facts, raw state, analysis snapshots, and critic
  world state are excluded;
- observer-invariant content is factored rather than repeatedly copied; and
- actor SharedObs, team-observation critic input, and privileged world-state
  critic input remain separate versioned contracts.

Replay and evaluation records store base observations once, together with the
source-axis/provenance mapping, any information-availability input not
losslessly derivable from those observations, `execution_information_mode`,
and the actor-input projection version. They do not persist a second copy of
materialized SharedObs tensors.

The Milestone 6 V1 frame schema supports both information regimes without
implementing the future compositor. Its availability field is an optional,
explicit Boolean matrix with axes
`(recipient_global_slot, sensor_source_global_slot)` and shape `(10, 10)`.
Conditional validation requires that matrix for `shared_obs` and forbids it
for `no_shared_obs`. Its diagonal, cross-team cells, inactive-recipient rows,
and inactive-source columns are false. Neither regime stores a materialized
SharedObs tensor.

## A2. Metric architecture and simulator cost

**Classification:** required architectural clarification.
**Supersedes:** R24 and Sections 2.6.16, 2.12.10, 2.13.18, 2.16.15, and
2.17.8 wherever they imply cumulative combat metrics or a full metric suite in
the simulator or ordinary training loop.

MARL-BattleGrounds uses three deliberately separate data planes:

1. The JAX simulator emits only fixed-shape authoritative facts for causes or
   resolved phase outcomes that would otherwise disappear.
2. Opt-in evaluation and scenario capture produces a complete semantic
   trajectory with `T + 1` frames and `T` transitions.
3. Trainer-owned telemetry contains algorithm-specific values such as policy
   statistics, optimizer state, runtime timing, and optional reward-shaping
   components.

Milestone 6 CP2 normalizes those planes through the following accepted host
boundary:

- `EvaluationEpisodeContextV1` owns the roster's fixed-slot identity/topology
  fields and the resolved configuration's per-slot body radius, movement,
  interaction ranges, maximum health, and recovery mechanics. It also owns
  policy, catalog, seed, `execution_information_mode`, `actor_projection`,
  `critic_information_regime`, `canonical_reward_mode`,
  `shaping_configuration`, and code provenance. It carries exactly ten
  discriminated policy-assignment rows; inactive slots use an explicit
  `not_applicable` variant.
- `GlobalAnalysisSnapshotV1` contains dynamic state only. It does not repeat
  configured-active, team, class, identity, body radius, maximum health, or
  any other static context/catalog value.
- Submitted and accepted actions occur exactly once, inside
  `TransitionFactsV1.action_acceptance_facts`. `EvaluationTransitionV1` may
  expose read-only conveniences but must not serialize a second action
  authority. `TransitionFactsV1` mirrors every core subtree and leaf name
  exactly rather than introducing shortened host aliases.
- Transition rewards are named `canonical_reward_by_agent` with fixed length
  ten and optional `canonical_reward_by_team` with fixed length two. Shaping
  values are excluded; only the shaping configuration identity belongs in
  context.
- A core recipient sentinel of `-1` normalizes to JSON `null`, agrees with the
  corresponding `has_recipient` flag in both directions, and reverses
  losslessly.
- Sparse events form an exactly 21-variant discriminated V1 union derived from
  facts and adjacent frames. `AgentDiedEventV1` records the newly dead recipient;
  one `LethalDamageContributionEventV1` separately records each authoritative
  positive source contribution on that lethal transition. Rank 90 orders the
  death event before its contribution events. Neither record claims a killer,
  last hit, or complete historical elimination credit. Aura attachments name
  direct transition-start covering emitters, not causal credit.
- Rank 120 uses family-specific canonical coordinates: a team-wave event sorts
  by `(120, team_index, -1, wave_subtype, neutral_source)`, while each realized
  agent respawn sorts by `(120, configured_team_index, agent_global_slot,
  respawn_subtype, neutral_source)`. This groups teams and places each wave
  before that team's realized agents.
- Catalog digests use finite-only canonical JSON with ASCII identifiers,
  recursive `-0.0` normalization, sorted keys, compact separators, and UTF-8
  encoding. Every code revision requires `source_tree_digest`; additionally,
  `dirty_patch_digest` is required exactly when `is_dirty` is true. Local paths
  are never durable policy or code identities.
- One public semantic validator consumes the context, transition-start frame,
  transition, and successor frame together, then re-decodes and exactly
  compares the deterministic event sequence. Structural model validation is
  not misrepresented as cross-record semantic validation.

Milestone 6 CP3 adds the following host-only streaming boundary:

- `validate_initial_evaluation_frame_v1` separately validates context join,
  artifact index zero, canonical frame identity, information-regime rules, and
  inactive dynamic padding. Artifact frame zero may correspond to any
  nonnegative simulator epoch; capture indices are never inferred from an
  arbitrary simulator step.
- An opt-in `EvaluationEpisodeObserverV1` accepts exactly one initial frame and
  then gap-free coherent context/start-frame/transition/successor-frame views.
  It tracks validated and successfully processed transition counts separately.
- Versioned reducers are immutable and copy-on-write. All reducer replacements
  and final rows validate before one atomic observer commit. A failed append,
  reducer, or final report poisons the observer, preserves the last validated
  prefix, and never publishes a partially updated statistic set.
- `EvaluationEpisodeCompletionV1` records only rollout completion
  (`complete`, `partial`, `interrupted`, or `failed`) and its authoritative
  basis or failure origin. `EvaluationProcessingStatusV1` independently records
  host processing success/failure. A processing failure after task terminal or
  declared horizon does not relabel the completed rollout as failed.
- Scientific endpoint observation is per statistic, using `not_applicable`,
  `observed`, `right_censored`, `competing_event`, or `unavailable`. It is not
  an episode completion state. Missing or ineligible data never become zero.
- Raw count, sum, numerator/denominator, duration, opportunity, and distribution
  components join one immutable episode context in
  `EvaluationMetricReportV1`. They retain exact subjects and dimensions but do
  not duplicate policy, seed, task/scenario, information-regime, reward, or
  shaping provenance on every row.
- An absent observer is the only disabled path. Explicit `training_light` and
  `debug` observers stream the standard semantic view without history;
  evaluation/scenario metric-complete observers retain exact `T + 1` frame /
  `T` transition prefixes. Debug adds no private-state payload.

CP3 adds no core callback, runner, logging sink, file I/O, replay artifact,
official metric formula, universal registry, materialized SharedObs tensor, or
private debug-state format. Replay persistence and loaded-file parity remain
the next milestone step.

The core mapping change permitted by CP2 is limited to a pure public
`core.axis_mappings` authority and bounded mechanical imports in the current
core environment/configuration consumers. Any changed mapping value, traced
operation, simulator result, or wider core edit stops the checkpoint for
review; host code never recreates an actor-relative indexing formula.

Derived totals, ratios, windows, engagement groupings, distributions,
leaderboards, ratings, and population summaries do not belong in `EnvState` or
the traced transition kernel. They are computed deterministically by host or
offline consumers. Ordinary training performs no host transfer, Pydantic
validation, event decoding, trajectory retention, or full metric calculation
unless an experiment explicitly enables a narrow diagnostic.

Researchers may construct optional JAX-native reward-shaping or auxiliary
reward components from ordinary transition outputs. Such components are
training interventions, not official evaluation metrics or canonical task
reward. A component using privileged facts must be labeled a privileged
training signal and must never become actor input.

## A3. Metric constitution and reporting surfaces

**Classification:** required scientific clarification.
**Supersedes:** Section 2.16 wherever a candidate is treated as official merely
because it is inexpensive or derivable.

An official metric must answer a real research question, be explainable in one
sentence, use genuine opportunities, preserve its raw sufficient components,
state its direction or descriptive status, avoid unsupported causal claims,
remain meaningful across algorithms and seeds, add material information, and
justify its implementation and cognitive cost.

The suite has separate surfaces:

- a compact primary team scorecard;
- a compact descriptive per-agent/per-class scorecard;
- an advanced long-form analysis export;
- quantitative controlled-scenario scorecards;
- diagnostics and quality-control outputs;
- cross-play, learning, runtime, and population summaries.

There is no global tactical score, opaque coordination score, or universal
agent-quality ranking. Presentation requirements such as sorting, tooltips,
quantiles, or CSV export do not create new simulator facts or primitive metric
columns.

Stable metric semantics are separate from evaluation-suite populations and
weights, experiment-manifest checkpoint/inference choices, result rows, and
presentation metadata. A paper may change an opponent pool or confidence
method without changing what a metric means or forcing a metric-version bump.

## A4. Combat terminology and attribution

**Classification:** corrective semantic clarification.
**Supersedes:** R24 and Sections 2.16.4, 2.16.5, 2.16.6, 2.16.7, and 2.16.8
where they use kills, last hits, K/D, solo kills, pentakills, source-level
effective-healing/overhealing credit, or ambiguous "effective" amounts.

The simulator records newly dead recipients and positive-damage contributors
on the lethal transition; it does not retain an unbounded damage history or
select a killer or last hitter. Official analysis therefore uses:

- **lethal-transition damage contribution:** an agent contributed positive
  recipient-modified gross damage on the transition where an enemy became
  dead;
- **lethal-transition contribution rate:** those contributions divided by the
  team's enemy deaths, with no-team-death episodes represented by a zero
  opportunity rather than a fabricated zero rate;
- **single-contributor lethal transition:** exactly one positive-damage source
  was recorded on that transition; this does not claim that the entire fight
  was a solo kill or that earlier damage did not matter; and
- **team wipe:** the task-defined complete elimination of a team, without
  assigning a pentakill.

K/D, last-hit kills, general elimination participation, solo kills,
pentakills, and arbitrary killer selection are not official metrics without a
separate versioned historical-contribution definition.

Every health-effect metric names its amount or health-resolution stage:

1. raw source output;
2. source-modified gross output;
3. recipient-modified gross effect entering simultaneous health resolution;
4. combat-resolution health after simultaneous resolution and clamping, before
   regeneration;
5. realized recipient net health change from transition-start health to that
   combat-resolution health; and
6. separately authored actual regeneration.

Recipient-modified gross damage may exceed remaining health. Simultaneous
damage, healing, and clamping do not support unique per-source attribution of
realized health loss or restoration. Overlapping equal-class auras and
anti-heal sources similarly support combined team/class effects, not invented
per-emitter causal credit.

## A5. Teamfights, engagements, and contextual behavior

**Classification:** corrective scientific clarification.
**Supersedes:** Sections 2.10.7, 2.12.11, 2.16.4–2.16.5, and 2.16.11, plus any
scenario or metric clause that assumes an observation-radius clique, team
centroid, out-of-combat timer, or casualty count is an authoritative teamfight
definition or that equates “single-shot” with one stochastic episode.

MARL-BattleGrounds does not define or roadmap a generic teamfight or engagement
detector, validator, or conditioned metric. All such candidates are rejected,
not deferred or validation-pending. Observation radius is a policy-information
mechanic rather than a fight boundary; a team centroid can merge separate
skirmishes; and the out-of-combat countdown is regeneration bookkeeping. No
core fact, host module, scenario validator, or milestone is reserved for
generic teamfight segmentation.

Peeling, kiting, flanking, body blocking, backline access, healing triage,
regrouping, rotations, escort/interception quality, trap discipline, and Burst
synchronization are evaluated through controlled quantitative scenarios rather
than vague episode-wide quality scores. A scenario has one primary quantitative
endpoint, at most two secondary margins, explicit violations, a horizon,
censoring semantics, frozen policy/configuration provenance, and replay
evidence.

"Single-shot" means that the evaluated policy does not adapt across attempts.
It does not mean that one stochastic episode constitutes an algorithm-level
sample.

## A6. Objective and class-specific metric corrections

**Classification:** required task-ownership clarification.
**Supersedes:** Sections 2.16.8–2.16.10 where formulas presuppose mechanics or
causal credit not yet owned by a task.

Task-specific metric formulas remain inactive until the owning task exposes
authoritative score, objective, reward, terminal, and event/snapshot truth.

In particular:

- KoTH control and contest denominators use eligible hill-timesteps, not bare
  episode horizon;
- per-agent hill occupancy or participation may be reported, but team score is
  not arbitrarily apportioned to agents;
- CTF class-specific carrying time is descriptive; there is no
  designer-preferred-carrier quality score;
- capture conversion uses eligible enemy-flag pickups and is always shown with
  that pickup exposure;
- task trades are reported through score/objective swing and casualty facts,
  not a universal teamfight-win label;
- Hunter Trap is an immediate targeted status/damage mechanic, so applications,
  control duration, lifecycle break rate, and follow-up associations replace
  the PDF's placed-trap uptime and trigger-rate language; and
- Priest Freedom application, uptime, and binding coverage are derivable, but
  exact counterfactual movement recovered is not an official metric.

## A7. Evaluation statistics and leakage control

**Classification:** required research-protocol correction.
**Supersedes:** Sections 2.15.17 and 2.16.16 wherever they prescribe a generic
three/five-seed target or leave “standard error or confidence interval”
unspecified.

Evaluation episodes estimate one trained checkpoint. Independent training
seeds—not episodes, teams, agents, deaths, or repeated matches—are the default
experimental units for algorithm-level claims. Results aggregate in this
order:

1. episodes within one homogeneous training-seed/evaluation cell;
2. evaluation cells under frozen predeclared weights;
3. independent training seeds with equal seed weight.

Rates pool raw numerator and denominator within a cell, but do not pool
opportunities across training seeds. Both teams from one match are paired;
agents are nested observations. Zero opportunities produce `N/A`, not zero.

Across frozen cells, opportunity metrics use the ratio of weighted raw
components, not the average of cell rates. Cell weights are not renormalized
around cells that happen to produce no opportunities. Independent training
runs receive equal weight, with seed-level results and uncertainty reported.

Official evaluation separates training, development/validation, and locked
test layouts, scenarios, opponents, partner pools, and seeds. Locked-test
results must not select metrics, tune predeclared analysis thresholds, choose checkpoints,
or shape rewards. Each experiment manifest predeclares its seed budget,
precision rule, checkpoint rule, endpoint hierarchy, uncertainty method, and
failure/missingness policy.

Learning/sample-efficiency results report both environment transitions and
active-agent decision transitions, plus wall-clock/compute provenance. A 1v1
and a 5v5 environment transition do not represent equal agent experience.

## A8. Milestone 6 Step 5 fact budget

**Classification:** efficiency correction to an unimplemented design.
**Supersedes:** Milestone 6 planning prose that approves 48 leaves / 1,677 raw
bytes or a `CooldownTransitionFacts` subtree.

The accepted pre-Step-5 baseline is 37 leaves / 897 raw bytes. Step 5 adds nine
leaves / 760 raw bytes:

- one `float32 (10,)` post-combat, pre-regeneration health stage;
- two `float32 (10, 2)` realized Charge- and ordinary-movement-phase
  displacement arrays;
- two `bool (10, 10)` aura emitter/beneficiary coverage relations; and
- four `bool (10, 9)` independent status-lifecycle cause matrices.

The resulting target is **46 leaves / 1,657 raw bytes**.

Cooldown start is already the accepted Ultimate action. Cooldown readiness is
the positive-to-zero change between adjacent semantic frames. Deriving either
does not recreate simulator rules, so permanent cooldown fact leaves would be
redundant. Host evaluation events may still present both lifecycle events from
those authoritative inputs.

Ordinary-movement displacement is algebraically redundant only in ideal real
arithmetic. In the actual `float32` trajectory, subtracting a rounded Charge
displacement from adjacent positions is not guaranteed to reproduce the
authoritative post-Charge boundary, can fabricate tiny nonzero sparse events,
and requires the host to import the rule that transition-start dead respawned
rows did not move. Retaining the 80-byte leaf is therefore an approved
phase-fidelity tradeoff: it preserves the exact phase-local result and keeps
respawn/liveness semantics in the simulator. It does not authorize a duplicate
geometry pass; the leaf must reuse the already resolved phase positions.

No accepted production core defect prompted this amendment. It removes
avoidable telemetry before implementation.

## A9. Spawn protection and out-of-combat regeneration

**Classification:** documentation of already accepted simulator drift.
**Supersedes:** R18, R19, Sections 2.5.8, 2.6.13, 2.7.9, 2.14.6, 2.14.7,
2.17.7, and any dependent wording that requires geometric spawn sanctuaries or
stationary-only regeneration.

Geometric spawn sanctuaries are replaced by deterministic team respawn waves
and timed spawn shielding. Shielded agents remain mobile and use the accepted
shield-specific collision, targeting, and visibility contract; the simulator
does not maintain a sanctuary region or sanctuary-egress subsystem.

Stationary-only regeneration is replaced by class-specific mobile
out-of-combat regeneration. Accepted positive combat interaction resets the
public countdown for its participants; an eligible living agent regenerates
according to its resolved class profile when the countdown permits it.
Movement is not itself a regeneration blocker. The authoritative transition
facts preserve countdown resets and actual realized regeneration rather than a
cumulative recovery metric.

## A10. Standard semantic replay ownership

**Classification:** required artifact-boundary clarification.
**Supersedes:** any historical wording that treats a raw simulator state,
renderer frame, or debugger session as the durable replay authority.

Milestone 6 version-1 replay stores the accepted evaluation context once plus
exactly `T + 1` semantic frames and `T` adjacent transitions, rollout
completion, independent evaluation-processing status, and a path-free
content-addressed metric-report reference. It does not store `EnvState`, policy
state, renderer summaries, local paths, or browser/session authority.

The episode context keeps exactly its eight CP2 schema bindings. The closed
replay/metric-report envelope has a separate exact binding map in the replay
header. Downstream scenario and actor-POV companions self-version and carry a
typed replay reference; their schemas are not retroactively inserted into the
replay header. A pre-link trajectory-content digest covers the header,
completion, processing status, frames, and transitions while excluding the
report reference and replay-level digests. The report artifact references that
pre-link content; the completed replay references the report artifact; later
scenario and actor-POV artifacts reference the completed replay. This one-way
graph prevents content-hash cycles.

Replay validity requires an explicit O(T) semantic pass using the public
initial-frame and four-record validators. Direct Pydantic revalidation proves
structure only. Offline analysis and presentation may consume only serialized
catalog mappings and captured semantic records; they never rerun a simulator
or recreate mechanics.

Canonical V1 persistence is finite local UTF-8 JSON with exact canonical-byte
equality after strict model and whole-replay validation. It rejects symlinks,
nonregular files, duplicate keys, non-finite or oversized/deep inputs, unknown
versions, and mismatched content references. Publication is atomic and
no-clobber: the metric-report object is durable before the referencing replay,
an existing report is reused only when bytes are identical, and replay targets
are never overwritten. The host loader owns frozen V1 wire dimensions and must
not import or initialize JAX, a backend, simulator, policy, or capture path.
The V1 filesystem backend fails closed unless POSIX directory-descriptor and
no-follow operations can prevent ancestor-symlink races; report reuse
synchronizes the exact compared file descriptor before a referencing replay is
published.

Scenario and actor-POV companions use the same bounded canonical JSON and
descriptor-bound, no-clobber publication contract. Scenario records are valid
only against their referenced replay and metric-report evidence. Exact
NoSharedObs POV exports use recipient-sliced schemas and keep submitted int32
intent distinct from category-bounded accepted actions. Their privacy claim is
defined over the recipient-content bytes; the outer artifact retains a truthful
completed-replay reference and may therefore differ when hidden source truth
differs.

Exact materialized SharedObs export remains unavailable until the Milestone 12
compositor exists. Milestone 6 may instead project a prominently labelled
`source_material_only` view containing the selected recipient's recorded base
sensor row and the recorded recipient-by-source availability inputs. That view
is not an actor-input artifact and may never be described as the composed
SharedObs tensor.

Offline presentation consumes these canonical records through a pure,
renderer-neutral scene projection. It may expose recorded durable state,
catalog mechanics, actor-relative mappings, and direct event evidence, but it
must not call simulator, geometry, visibility, masking, policy, or mechanic
helpers. Researcher and actor-authorized presentation roots remain
structurally distinct.

## A11. SharedObs recorded visual-union presentation

**Classification:** harmless clarification of the rendering and learner-input
boundary established by A1 and A10.
**Clarifies:** A1 and A10; the Milestone 6 presentation contract; and the
conceptual name `shared_obs_recorded_visual_union`.

`shared_obs_recorded_visual_union` names a rendering-only authorized view. Its
frozen V1 wire literals are `shared_obs_visual_union` for the observation mode
and `authorized_same_epoch_sensor_source_visual_union` for the construction
basis. These names describe the same presentation contract; existing V1 wire
literals are unchanged.

The view keeps the selected recipient's own recorded base-sensor row separate
and may add only recorded, same-decision-epoch sensor rows from same-team,
configured-active sources for which the recipient's recorded source
availability is true. It is an authorized visual union, not a recomputed
observation: presentation code must not recreate geometry, visibility, line of
sight, masks, mechanics, or simulator state.

The visual union excludes teammate action masks, prior-action or other history,
rewards, recurrent or policy state, transition facts, critic input, and hidden
Oracle state. The diagnostic `source_material_only` view described by A10 is
not a product Agent-POV presentation root and must not be installed or labelled
as one.

That exclusion governs the durable Agent scene and learner-facing material; it
does not make visible battlefield actions disappear from replay or debugger
playback.  The presentation layer may carry a separate, ephemeral visual-event
projection derived from the canonical incoming transition.  Every agent and
phase anchor in an emitted row must already be authorized by the recipient's
fog-filtered start or successor scene. A payload fact that describes or derives
from an endpoint requires that endpoint to be authorized even when its canonical
coordinate anchor belongs to an earlier phase; for example, hidden successor
health, regeneration, and cooldown outcomes cannot be disclosed through a
transition-start anchor. Hidden sources, targets, aura emitters, inactive
identities, and global-only pulses are omitted server-side. Surviving rows use a
dense recipient-local identity axis, so canonical event IDs, hidden counts,
ordering gaps, slots, and Oracle frame or transition identities never cross the
Agent presentation boundary. This projection exists only to render the same
visible action semantics as Oracle View under fog; it is not stored in or
derived from the learner's actor-input artifact.

Neither representation is a materialized SharedObs learner input. Exact
SharedObs actor-input export remains unavailable until the Milestone 12
compositor is implemented, performance-tested, and accepted. Presentation and
replay support therefore do not activate SharedObs training or authorize any
claim that the composed learner tensor is available.
