# MARL-BattleGrounds Specification Amendments

> **PROPOSED NORMATIVE CONTRACT — ACTIVATES ONLY AFTER EXPLICIT USER
> ACCEPTANCE.** Until activation, this draft records the intended disposition
> and does not silently supersede the design PDF.

This document proposes changes to
`MARL_BGs_Design_Document.pdf`. The PDF remains the original architectural
blueprint; after explicit acceptance, this file becomes the controlling public
source when the two disagree. An activated amendment changes only the clauses
it names. Unmentioned requirements remain in force.

The amendments below were drafted before Milestone 6 Step 5 implementation.
They deliberately favor the four project North Stars: researcher-centricity,
low sample complexity, meaningful tactical and strategic team behavior, and
professional MARL/software engineering.

## A1. Evaluation information regimes

**Classification:** risky, accepted design drift.
**Supersedes:** R20; Sections 2.4, 2.4.1, 2.4.8, 2.4.11, 2.6.18, 2.8.5,
2.13.8, 2.15.7, 2.16.2, and 2.16.19; Appendix A.11; and baseline or paper
clauses that make NoComm the default and SharedObs merely optional.

SharedObs is the approved future default execution-time actor-information
regime. It is not current runtime behavior and does not become active until its
versioned learner-input projection is implemented, performance-tested, and
accepted before canonical baseline training.

NoComm remains an official first-class regime. It must remain selectable and
must be evaluated and reported separately from SharedObs. Results from the two
regimes must never be pooled into one benchmark cell or summary.

The canonical implementation boundary is:

- `shared_obs` and `no_comm` are values of an extensible learner-side
  information-regime configuration;
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
- in `no_comm`, the learner returns the actor's base observation and bypasses
  material shared-bank and recipient-by-source-mask construction;
- teammate masks, previous-action/history fields, recurrent state, policy
  memory, rewards, transition facts, raw state, analysis snapshots, and critic
  world state are excluded;
- observer-invariant content is factored rather than repeatedly copied; and
- actor SharedObs, team-observation critic input, and privileged world-state
  critic input remain separate versioned contracts.

Replay and evaluation records store base observations once, together with the
source-axis/provenance mapping, any information-availability input not
losslessly derivable from those observations, the execution regime, and the
actor-input projection version. They do not persist a second copy of
materialized SharedObs tensors.

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
