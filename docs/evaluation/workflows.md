# Research Workflows

Start with ordinary environment calls. Add the evaluator, recorder or tournament
helper when that saves work. You own the method that chooses actions and learns;
MARL-BGs owns environment behavior and the shared measurement/recording rules.
The [environment example](../../examples/environment.py) and
[evaluation example](../../examples/evaluation.py) run the public interfaces.

This page describes implemented behavior. Later accepted M8 contracts, including
Systems, stage tracking, phase-specific map defaults, fixed-team spawn-paired
evaluation and official snapshot reuse, are not all implemented by the current
raw-environment packet. Do not assume a proposed call exists until its packet is
qualified. In particular, current `evaluate(..., phase="validation")` still needs
explicit validation maps, and the current generic tournament pairs team roles.

## Map ID Change — 2026-09-12

Public map IDs now group maps by their use. Map geometry and game rules are
unchanged.

| Public IDs | Maps |
| --- | --- |
| 0–11 | Curriculum maps, in their existing order |
| 12–41 | Other training maps, in their previous ID order |
| 42–46 | Validation maps, in their previous ID order |
| 47–51 | Test maps, in their previous ID order |

Training still includes all 42 maps with IDs 0–41. Each map's public ID, numbered
name, authored asset ID and saved-map folder now use the same number. For example,
old IDs 40, 0 and 20 are now 0, 12 and 48 respectively. Three Body Problem is now
`tdm_map_id_48_three_body_problem_test`. Its familiar name, geometry and split
stay the same. All scenario content is unchanged.

The renamed authored files have new byte hashes. Their geometry hashes and
packaged geometry bytes stay unchanged. The package catalogue records the new
source names, paths and byte hashes. Use `list_tdm_maps()` to discover current IDs
and source details. Restart DevClient and reopen saved drafts after this change
so its selections use the renamed assets.

Existing recordings and result files keep their recorded IDs, names and content.
Saved configurations containing only numeric map IDs need an explicit old-to-new
translation using the original catalogue, or map selection through current
discovery. Do not treat an old number as the new map with that number. Resume or
reproduce an older run with its original code and catalogue.

Custom tournaments sort maps by public ID. If a map selection spans the groups
above, its new order can change which seed each map receives. Use the original
catalogue and schedule when reproducing an older tournament. Historical
measurements below retain their original labels and source hashes.

## Evaluate and inspect

### Policy Inputs

`Policy.apply(variables, carry, actor_input, action_mask, key)` receives one
actor's permitted view. `actor_input` contains `observation`, `source_bank` and
`source_availability`. It has no simulator team ID or global slot number.

Use `observation.self_features` for the actor's own class and current state.
A shared network can use those features to choose different actions for different
classes. `observation.self_ally_index` is a scalar integer from 0 to 4 that locates
self among the five ally rows. It is a lookup aid; the adapter does not add it
to a network's feature vector. It stays fixed through death and respawn.
Inactive padding uses zero, distinguished by the existing activity mask.

Unit rows still have 58 columns. Column 3 is now `is_enemy`: zero for self and
allies, one for a visible enemy. Hidden enemy rows are still all zero. Use the
ally/enemy grouping and visibility/activity masks to distinguish hidden enemies,
allies and unused positions. The flag alone cannot make that distinction.

Each SharedObs actor receives five own-team source positions. Source features
have shape `(5, 10, 58)`, visibility `(5, 10)`, objectives `(5, 8, 12)` and
availability `(5,)`. Candidate rows are five allies followed by five enemies.
Self is an unavailable shared source because its own observation arrives
separately. All unavailable source rows are cleared before the policy call.
Dead teammates remain authorized but provide no sensor material.

The direct SharedObs callback takes five arguments: observation, mask, key,
source bank and availability. The old final global-slot argument is removed.
The direct NoSharedObs callback still takes observation, mask and key. Routing
and result files retain physical Team A/Team B and global agent identities;
those metadata fields are outside method inputs. A normal `evaluate` call still
assigns its first policy to Team A and its opponent to Team B.

Custom source subsets belong in the wrapper state before actions are chosen:

```python
# chosen_sources is a Boolean (..., 10, 10) subset of the default availability.
state = state._replace(source_availability=chosen_sources)
observations = env.get_observations(state)
```

Pass these compact observations to `apply_policies` as usual, then pass the same
state to `env.step`. Step keeps the chosen subset. You may change it before the
next decision. A partial reset restores default availability only in reset lanes.
Automatic replay capture uses the same state field, including when the subset
changes across recorded chunks. Changing only a detached observation object
would leave the recorder unaware of that change; keep the state as the source
of truth. These are routing inputs; actor callbacks still receive only their
five local source entries. Explicit low-level capture remains available.

Current recordings version these changed inputs explicitly. Historical
recordings retain their original team-ID feature meaning and remain readable.
Run files record the current actor projection identities. Resuming with a missing
or different input contract is rejected before any saved table is changed.
An unchanged feature width does not make an old checkpoint compatible: reproduce
old results with their original code, and revalidate checkpoints under the new
input contract. This change makes no new learning or slot-fairness claim.

### Run Evaluation

Run four complete ALPHA/BETA episodes and save full measurements plus two replays:

```bash
JAX_PLATFORMS=cuda .venv/bin/python examples/evaluation.py evaluate \
  --episodes 4 --num-envs 4 --metrics full --save-replays 2 \
  --output-dir runs/evaluation
```

`--episodes` is the total evaluation budget, cycled across the chosen maps in
order. Four episodes cover four of five maps; use a multiple of five for equal
coverage of that split. The executor batch changes concurrency, not schedule
identities. Compilation affects the first call. The command prints the unique
run directory and saved paths. CPU execution remains available for correctness
checks; GPU JAX is the performance target.

Open a selected replay with its printed path:

```bash
.venv/bin/python scripts/dev/replay_viewer.py --replay PATH_TO_REPLAY
```

The Viewer reads saved evidence; it does not run the policy again. **Up to Current
Tick** analyzes the selected prefix and **Entire Episode** analyzes the recorded
endpoint, which may itself be incomplete. CSV exports are numeric measurements.
**Episode Details** exports context/configuration/completion/runtime metadata,
without trajectory arrays. Replay export and full metric collection are separate.

Without `output_dir`, `resume_from` or a supplied writer, evaluation creates no files. For optional pandas analysis:

```python
import pandas as pd
import marl_battlegrounds as marl_bgs

result = marl_bgs.evaluate("tdm-alpha", "tdm-beta", num_episodes=100, metrics="full")
episodes = pd.DataFrame(result.full_metrics)
print(episodes["team_a_score"].mean())
```

Pandas is not a runtime dependency. For a saved run, read the path in
`result.paths["full_metrics"]`; the evaluator does not also retain those saved
full rows in memory. Read selected columns with `usecols` when a full table is
unnecessarily large. Blank cells mean unavailable measurements, not zero.

Current scalar schema 13 has 26 priority measurements and 11,158 full measurements,
including priority. A full run CSV adds 30 identity columns; a Viewer CSV adds 49.
The accepted future 16-measurement priority migration belongs to a later packet.
Existing files retain their headers; incompatible append/resume is rejected.
Replay format versions and analysis schema versions are separate.

The [metric specification](metric_specification.md) owns formulas, applicability,
zero/blank rules and display mappings. The [column dictionary](metric_columns.csv)
provides exact CSV names and positions. In the Viewer, choose a topic and its
**Totals** or **By Recipient** view. **Find a Measurement** accepts plain words,
partial words, agent labels and exact CSV names. Arrow keys select a result,
Enter opens it and Escape closes the list. Hover/focus help names its CSV column,
who acted, who was affected, numerator, denominator and blank-value rule.

Tables hide structurally impossible rows using the recorded active roster;
the fixed CSV still contains every column. A valid zero or an undefined fraction
does not make a row structurally impossible. Amounts/counts display up to two
decimal places and fractions/ratios up to four; CSV retains stored precision.
Viewer identity includes replay and analysis digests, scope, local frame index,
simulator step and recorded roster/policy identity. Local frame zero can start
at a later simulator step. Run CSV identity instead names run, phase, pass,
episode, seed, map, configuration and policies/checkpoints.

## Choose metrics and replays independently

`metrics="priority"` is the default. `none` skips optional measurements, while
`full` collects the complete scalar set. Explicit full/replay selections are
finite Python integer iterables of episode IDs, not a sampling frequency:

```python
import marl_battlegrounds as marl_bgs

env = marl_bgs.make(
    "tdm", map_id=0, num_envs=128, metrics="priority",
    full_metrics_episodes=range(1000, 50_001, 1000),
    replay_episodes=range(49_951, 50_001),
)
```

Selectors are prepared during setup. Numerical membership stays on device;
there is no per-transition Python lookup or file write. IDs not reached by a
run produce no capture. Automatic reset allocation may leave unused IDs, so a
numeric range does not promise a particular number of completed games. Metrics
consume authoritative transition facts without needing a replay or second game.

## Train and record

The basic explicit-reset workflow needs no researcher-written spawn or ID logic:

```python
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=0, num_envs=128)
reset_key, action_key, step_key, next_reset_key = jax.random.split(jax.random.key(42), 4)
obs, state = env.reset(reset_key)
actions = env.sample_actions(action_key, state)
obs, state, reward, done, info = env.step(step_key, state, actions)
# Keep any terminal transition before replacing its finished lane.
obs, state = env.reset_done(next_reset_key, state)
```

Map defaults use canonical rosters and split an even native batch equally between
the source's complete spawn banks and their exchange. Teams, supplied roster
order, agent slots and world directions stay fixed. Scalar and odd-sized map batches must
set `balance_spawn_locations=False`. Exact `env_config=` bypasses automatic
preparation without another opt-out. An unconfigured `make("tdm")` still works
when reset receives its first configuration.

For custom preparation, call `balanced_spawn_configs(env_config, num_envs=...)`
once on the immutable source or selected sources. Reset consumes those exact
values. Do not prepare `state.config` again: it already contains resolved banks.
Both complete banks determine later respawn locations too.

Reset chooses explicit `env_config`, then a supplied state's retained config,
then constructor defaults. `reset_done` replaces only finished lanes. Advanced
reset supports `state=`, `reset_mask=`, `initial=` and explicit `episode_id=`.
Authored starts stay exact. The public keyword is `env_config`, including in
`EpisodeSpec`; positional meaning, Core `state.config` and saved keys stay intact.

Native batches initially allocate IDs 1 through B and reserve a fresh B-ID block
when any lane resets. Gaps are valid. External `vmap` of independent scalar
contexts starts each at ID 1: use explicit globally distinct IDs or separate
passes before combining recordings. Exact evaluation schedules own their IDs.
Checked int32 counters/IDs stop being valid at overflow and set persistent
`state.lifecycle_error`; they do not silently wrap. Real counts include terminal
transitions and exclude reset calls and terminal padding.

Equal lane counts alone do not prove balanced training. A supported training
stage must consume equal real advances from each spawn choice; equal completed
game counts are insufficient. This raw API exposes the underlying lifecycle,
while library stage tracking and coordinated recorded-training recovery belong
to later packets. Do not claim those later checks from this example.

Scalar/native calls compose with `jit`, external `vmap` and `lax.scan`. Reuse a
callable and pass changing numerical data dynamically; repeatedly making new
jitted bound functions does not promise reuse. `scenario=` is a host shortcut;
prepare an `initial=` snapshot before compiled loops. `env.agents` names capacity
slots, while `agent_info(state)` describes active rosters. Spaces describe
structure; masks describe current legality. `sample_actions` respects coupled
target/Ultimate choices without constructing unnecessary SharedObs input.

Your learner owns its update and logging decisions. For optional recording,
create `RunWriter(..., phase="training", pass_id=...)` and send every returned
`info`, or the complete stacked `infos` from a scan, to `writer.write` on the
host. Sending only the final scan step loses earlier packets/completions. Bind
recorded policy/configuration context through the writer's documented methods.
Training episodes spanning updates must be labelled as evolving policies,
not assigned falsely to a single checkpoint.

Full info arrays are dense even on nonterminal steps or sparse full selection.
Schema 13 returns 55,790 logical bytes of values/validity per full-result lane;
retaining 1,024 lanes over 128 steps is about 6.81 GiB for that subtree alone.
This is a size calculation, not peak-memory measurement. Priority without full
selection has no full subtree. Empty replay selection has no capture subtree;
selected scan packets still cost chunk memory. Use bounded chunks. The writer
spools incomplete replay data and publishes completed artifacts.

The writer defaults to a 128-row buffer. A larger buffer can reduce repeated
metadata writes but uses more RAM and can leave more completions to repeat after
abrupt interruption. Full rows are much wider than priority rows. `flush()` and
normal context-manager exit establish durability and report write failures.
See the [replay contract](replay_format.md) for versioned persistence behavior.

## Repeat validation in one run

```bash
JAX_PLATFORMS=cuda .venv/bin/python examples/evaluation.py validation \
  --episodes 10 --num-envs 4 --metrics full --output-dir runs/validation
```

This runnable example explicitly discovers validation maps and shares a writer
across two differently named passes. The current `phase` label does not choose
the split for you. Freeze the checkpoint and pass explicit maps; use a new
`pass_id` for each selection point. A `Policy` carries the apply function,
variables, initial recurrent carry and optional checkpoint identity. Each actor
receives only its authorized input and mask. Validation results must follow the
predeclared selection rule and must not train on held-out evaluation evidence.

## Run a tournament

```bash
JAX_PLATFORMS=cuda .venv/bin/python examples/evaluation.py tournament \
  --episodes 20 --num-envs 8 --output-dir runs/tournament
```

Here the budget is per unordered policy pair, covering every chosen map and both
current team assignments. Twenty games across five maps gives two complete
paired blocks per map. This is a workflow check, not enough evidence for a broad
scientific claim. The current generic default of 100 is not the final official
snapshot budget; that official number remains undecided.

Current ratings center on 1,200. The shared fitter and 5,000 matched bootstrap
resamples run after complete coverage. Missing matches or failed fits are errors;
insufficient uncertainty evidence stays unavailable. Required outcomes support
rankings and matchup/map summaries even with optional metrics disabled.

The accepted official design is a monthly frozen Big 12 snapshot, optionally
with one challenger. The wider baseline library can grow independently. Verified
reuse, uniform budget overrides, promotion rules and the automatic headline
report require their later implementation/qualification packets. Local generic
results do not admit or publish an entrant. See the
[protocol](protocol.md#big-12-tournament-and-baseline-library) for scientific rules.

## Resume and combine results

A new `output_dir` call creates a unique child run directory. Use
`resume_from=run_dir` for explicit compatible continuation. Preserve recorded
policies, maps, seeds and capture settings; a changed execution batch/chunk size
does not change schedule identity. Durable completed episodes are skipped and
interrupted suffixes recovered under the existing writer contract. Failures name
the run and are recorded when storage remains usable. This evaluator recovery is
separate from future coordinated learner-checkpoint/writer restart.

Episode IDs are local to a pass. Join with `run_id`, `phase`, `pass_id` and
`episode_id`, plus the recorded participant ownership. Choose means, medians or
pooled count ratios deliberately; they answer different questions. Required
tournament population weights belong to its shared statistics authority.

## Historical Reporting and Viewer Evidence

The following retained notes record schema changes, exact comparisons and
Viewer evidence from their original revisions. Preserve their numbers and raw
references. They do not establish a new performance result for the present
source. Current formulas and navigation are owned by the metric specification.
Code fragments in this historical record assume the analysis variables above.

Schema 8 changed column order. Schema 9 changed the healing term to **Excess**
in column names, the dictionary and the viewer. Schema 10 removed 46 observed
respawn-wait columns. **Respawning** shows each
team's wave count, mean agents returned per wave, and mean waiting ticks. The
wave count includes scheduled waves when nobody returns. The mean agents per
wave includes those empty waves: one wave with no returning agents gives a mean
of zero. Before any wave, that mean is blank. The
mean wait counts only time seen in this recording, including waits already
underway at its start or still open at its end. Death counts and time dead
stay in **Deaths and Time Dead**. Earlier CSV files keep their original columns.
Surviving names, values and blank-value rules stay the same. This cleanup makes
no new speed claim.
The schema-10 check matched all 11,100 retained measurements by name across five
recorded boundaries: 55,500 values and their blank-value flags stayed exactly
the same. Two separate reviews agreed on the 46 removals. Browser checks covered
the six Respawning rows, their tooltips in both POVs, and matching CSV values.
This change did not rerun or replace the performance measurements below.

Schema 11 added 40 recipient measurements from counters already collected:
Trap periods, the fraction broken by damage, mean time left when broken, and
chances to save each agent with Priest healing. Team measurements stay intact.
The recipient can be any active class; a rescue opportunity requires a Priest
on that recipient's team. Initial Traps remain measurable without a Hunter.
No new per-tick counters or counter updates were added. The wider final result and
CSV contain more values; this update makes no new speed claim.

These columns do not repeat the meaning of an existing column. The ten Trap
break rates are calculated conveniences: each agent's broken Traps as a share
of its Trap periods. Counts and rates are both kept for direct analysis.
Two Hunters can make two Trap activations on one tick but create one Trap
period. A rescue chance can be present even when no Priest actually makes the
save. Team totals alone cannot show which agent these events concerned.

The schema-11 checks preserved all 11,100 earlier measurements at five public
trajectory boundaries: 55,500 values and their blank-value flags match exactly.
Eight CPU/GPU correctness cases also pass, including simultaneous Hunter
activations and combined Priest healing. The collector's 61 counter arrays stay
unchanged. See `artifacts/m8-ultimate-topic-fixes/` for the raw comparisons and
independent reviews. These checks do not repeat the performance matrix below.

Schema 12 added ten `agent_i_basic_rescue_participation` columns and two
`team_{a,b}_basic_rescue_fraction` columns beside the Basic save counts.
An agent's fraction is its Basic save contributions divided by all unique saves
by its team, from any ability. A team's fraction counts each save helped by
Basic healing once, then divides by all its unique saves, from any ability.
If two Priests help with the team's only save, each can have participation
`1.0` while the team still has only one save. Basic and Ultimate healing can
both help with that same save, so their fractions can overlap. These fractions
are blank when the team has no saves; inactive agents' shares are also blank.
They count actual saves, not unused chances to save someone. See the
[Basic healing-save definitions](metric_specification.md#schema-12-basic-healing-save-shares).

The twelve shares reuse existing counts when the full result is built. They
add no per-tick counters or counter updates. All 11,140 schema-11 column
names and values remain. Clearer display names, explanations and row order
are separate from these twelve added measurements. They change no simulation
rule, action, observation or reward, and establish no new performance result.

The schema-12 check compared all 11,140 earlier measurements at five public
trajectory boundaries. All 55,700 values and their blank-value flags matched
exactly, and all 61 counter arrays stayed unchanged. Focused cases for the
Basic save shares passed on CPU and GPU, including shared saves, repeated
Priests and zero denominators. These are correctness checks, not a new speed
measurement. The saved evidence is in `artifacts/m8-six-section-fixes/`.

Schema 13 adds `team_{a,b}_warrior_charge_applications`,
`team_{a,b}_rogue_poison_applications` and
`team_{a,b}_priest_holy_word_salvation_applications`: one column per ability and
team, six columns in total. Each counts allowed activations, including repeat
uses and entirely excess Salvation healing. The totals reuse existing activation
counts; no per-tick counter is added. A team with an active agent of that class
and no uses has a real zero. A team with no active agent of that class has a blank.
The new team columns appear only in their named Ultimate topics. Agent activation
rows remain available.

No existing CSV column is renamed. All Slow, Stun and Anti-Heal columns keep
their exact names, meanings and Status Applications rows. The new Charge and
Poison ability columns are separate from those effect measurements, even when
their values are equal. Mage Burst and Hunter Trap keep their existing team
ability application columns.
See the [schema-13 contract](metric_specification.md#schema-13-team-ability-application-columns).

The mapping audit checked exact CSV names and positions in all five ability
tables across five roster layouts. Its 157 checks passed, including every
earlier definition and the retained team status rows. Earlier checks of equal
numbers did not establish that name contract. Charge and Poison needed separate
ability columns, and Priest needed team Salvation counts. A separate
ability-count search issue could select the wrong kind of row. The name audit
is in `artifacts/m8-search-direction/ultimate-team-column-audit.json`; it does
not qualify numerical replay values or browser behavior.

Separate checks passed 718 search questions and all 55,790 exact CSV lookups
across five recorded rosters. The 11,152 earlier values and their blank-value
flags matched exactly in 18 before/after reducer cases. Both browser checks
passed, including same-boundary CSV values, all 43 tables in both POVs, narrow
tooltips and keyboard navigation. The saved reports and screenshots are in
`artifacts/m8-search-direction/`.

On the current machine, matching a question took a median 0.34 ms and a maximum
21.71 ms. Twenty warmed browser inputs showed their results in 24.4–28.0 ms
(median 26.4 ms), measured through two animation frames. Building the cached
search index took 242.23 ms once. These measurements cover search only. They
do not measure simulation, GPU rollouts or learner speed, and do not replace
the retained evaluation performance results below.

Ultimate tables use Mage Burst, Warrior Charge, Hunter Trap, Rogue Poison and
Priest Salvation in local row and tooltip wording, with each row's exact CSV link.
Other topics retain their shared display names. Shared death and save totals explain when
they supply the all-ability denominator for a nearby participation fraction.
Burst has no displayed kill fraction using Total Kills, so it omits that shared
row. Total Kills stays in Episode Results and the CSV. A share of team Ultimate
damage can include several classes. Effects on Team A describe what Team A's agents experienced,
even when Team B applied the effects.

The schema-8 checks compared all 11,146 old and new definitions by column name.
Every number and blank-value flag also matched exactly across all 188 boundaries
of one saved Cheshire Cat replay: 2,095,448 entries. Separate checks compared
100,314 row-applicability decisions across nine rosters, including moved classes,
repeated classes and inactive slots. This supports preservation for the checked
cases; one recording cannot prove every possible game. Five GPU correctness
cases also checked selected full collection, partial reset, contribution credit
and healing precision. No speed benchmark was repeated for the navigation change.

The schema-8 browser checks covered all 43 tables, both POVs, and CSV equality at current
and entire-episode scope. Search reuses one catalog request across topic, scope,
seek and POV changes. The independent review and raw check results are in
`artifacts/m8-metric-navigation/`; start with `navigation-audit-review.json` and
`values-comparison.json`. There are 39 nonempty primary CSV ranges: the Charge
and Salvation tables reuse columns whose primary homes are elsewhere.

Before schema 10, the class-filter audit independently reviewed all 11,146 columns and
resolved every disagreement between two reviewers. The final shared list covers
858,242 display decisions across 77 distinct rosters, including moved classes,
repeated classes, missing classes and inactive slots. Both reviewers checked the
code against that list with no differences. CSV data and numerical rules stay
unchanged. These are roster checks, not extra games or a speed benchmark. The
original reviews and resolved decisions are in
`artifacts/m8-class-filter-audit/all-roles-consensus.json`.

For that historical Cheshire Cat recording and catalog, the separate search response contained
14,507,239 bytes of definitions and applicability information. It is loaded once
and kept for that replay. This is extra host/browser storage and transfer, not
new work inside the JAX simulation. It is not a measurement of peak browser RAM.

Every number shown in the viewer has a CSV column. Hover over a measurement,
or focus it with the keyboard, to find **CSV Column** and copy its exact name.
The viewer shows amounts and counts with at most two decimal places. Fractions
and ratios keep up to four. CSV exports keep the full stored values. Adding the
same float32 values in different orders can leave tiny differences in the last
digits; rounding the display does not change the recorded measurements.
The help text says who did what. **From** is who acted; **To** is who was
affected. **Numerator** names the amount being compared; **Denominator** names
the total it is compared with. Team explanations name Team A or Team B.
**Blank When** explains why there may be no number. For example, four Basic
ability activations out of ten give `0.4`, or 40%. An agent that activated no
Basic abilities has no Basic allocation fraction to calculate, so that
fraction is blank.

Allocation means the share of one agent's output that went to one target.
Contribution compares that agent's output with its team's output to the same
target. For example, `agent_5_to_agent_3_burst_damage_allocation_fraction` of
`0.4336` means 43.36% of Agent 5's damage while Burst was active went to Agent 3.
It does not mean Agent 5 supplied 43.36% of all damage Agent 3 received. Read
the named ability, effect and time rule in both the Numerator and Denominator.
For harmful-effect measurements, the recipient must already have that named
effect at the start of the tick. Applying it later that tick does not count.

Effective Priest healing means delivered healing minus healing above the
health cap after that tick's damage. It can offset damage even when health
does not rise or the ally still dies. The separate healing-save measurements
require the ally to survive damage that would otherwise have killed it.
**Total Effective Healing Received** also includes regeneration, so its
guidance is context dependent, just like **Regenerated Healing**. Pure
effective Priest-healing amounts keep their existing guidance. Amounts stay
beside their fractions; received healing stays separate from supplied healing.

Read a column name from left to right:
`team_b_to_agent_1_basic_applications` counts how many times Team B used Basic
abilities on Agent 1. `agent_5_to_agent_1_basic_applications` counts how many
times Agent 5 did so. Attempts rejected by the game do not count. The viewer
may call these uses **received from** Team B or Agent 5; each number still has
just one CSV column. Formation uses `agent_0_and_agent_1_` for the two teammates'
distance apart. See the
[schema-3-to-4 naming rules](metric_specification.md#reading-source-and-recipient-names).

Schema 5 removed 206 approved columns and grouped the remaining columns more
clearly. Schema 6 added 12 Basic kill fractions. Schema 7 adds 20 columns for
class Basic kill counts and fractions. Within each subject, All abilities come
before Basic and then Ultimate, with each amount followed by its fractions.
Surviving names and calculations stay the same. The
[column dictionary](metric_columns.csv) uses the same
descriptions, numerators and denominators as the viewer.

`agent_i_basic_kill_participation` is the share of its team's kills that Agent `i`
helped with its Basic ability. `team_a_basic_kill_fraction` and
`team_b_basic_kill_fraction` are the share of each team's kills that had at least
one Basic helper. Credit needs Basic damage on the tick the enemy dies, or useful
Priest Basic healing of a teammate who damaged that enemy on that tick. A Mage
Basic attack during Burst remains Basic help.

Two Basic helpers can each get credit for the same kill, but the team counts it
only once. Basic and Ultimate help can overlap: if both help with the team's only
kill, both team fractions are `1.0`. They do not add up to the number of kills.
These fractions are blank when that team has no kills. Inactive agent fractions
are also blank. See the [schema-6 definitions](metric_specification.md#schema-6-additions-and-migration).

Class Basic kills use names such as `team_a_mage_basic_kills` and
`team_a_mage_basic_kill_fraction`. All five classes have these two columns on
both teams. The count includes each enemy death once, even if several agents
of that class helped with Basic abilities. The fraction uses this same count:
**Numerator: Kills helped by that class's Basic abilities. Denominator: All kills
by the named team.**

If two Mages help with Basic attacks on one kill, the Mage count is one. If a
Mage and a Warrior each help on that kill, both class counts are one. These
class fractions can overlap with each other and with Ultimate help. Useful
same-tick Priest Basic healing counts; healing that is all excess does not. Mage
Basic attacks during Burst count too.

A class present in an active slot has a real zero count if it has no Basic kill
credit. Its fraction is zero when the team has kills, and blank when the team
has none. If the class has no active agent on the team, both columns are blank.
See the [schema-7 definitions](metric_specification.md#schema-7-class-basic-kills-and-migration).

The existing single- and multiple-contributor kill counts and fractions appear
in both **Kill Contributions** and **Team Coordination**. Both groups show the
same values from the same CSV columns; no duplicate columns are added.
Only damage and useful Priest healing on the enemy's death tick count as help.
If Agent 0 damages an enemy on tick 1 and Agent 1 kills it alone on tick 2,
the kill has one contributor. If a Priest also gives Agent 1 useful healing
on tick 2, it has two. Healing that is all excess earns no kill credit.

The viewer uses the recording's active slots, classes and ability target rules
to omit structurally impossible detail. The CSV keeps every column across roster
changes. A zero value, unused ability or undefined fraction is not a reason to
hide an otherwise applicable measurement. Recipient-owned initial statuses remain
relevant even when the recording has no current caster of that class.

Download names identify the episode, scalar schema, scope, and frame. For a focused
analysis, pandas can read only the columns needed:

```python
episodes = pd.read_csv(
    result.paths["full_metrics"],
    usecols=["episode_id", "agent_3_to_agent_8_ultimate_application_fraction"],
)
print(episodes["agent_3_to_agent_8_ultimate_application_fraction"].mean())
```

This example averages the per-episode fraction of Agent 3's Ultimate ability
activations aimed at Agent 8. Using the summed counts instead weights by the
number of activations. Both counts are exported; researchers choose the summary
that answers their question. Reading a subset saves analysis memory without
changing collection or the original CSV.

## Performance qualification

**GPU execution with JAX is the performance target.** Measure compilation and
reuse, synchronized execution, peak GPU memory, transfers, and the setup and
recording work needed by the complete GPU workflow. Host work still matters
when it makes that workflow slower or uses extra memory. Separate CPU rollout
speed is not an acceptance requirement. CPU tests check correctness and
compatibility; their elapsed times help maintain the test shards, not compare
simulator performance.

Future GPU environment batches are **32, 64, 128, 512 and 1024 only**. Follow
the [shared GPU efficiency protocol](../dev/gpu_sanity.md#gpu-efficiency-protocol)
for realistic play, timing, memory and complete-workflow comparisons. The
report must include absolute speed and resource use as well as relative changes;
profile remaining costs before calling the changed path efficient. The
following existing foundations profile is a **short-episode reset stress test**
with fixed neutral actions. It measures that focused workload, not normal combat
or learning throughput. Its explicit sizes follow the new rule; older script
defaults and GPU correctness fixtures still need alignment before their next use.

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m scripts.dev.benchmark_evaluation \
  --foundations --backend gpu --api automatic --sizes 32 64 128 512 1024 \
  --lengths 16 128 --rollout-modes none priority --repeats 5 \
  --output artifacts/m8-api-foundations-gpu
```

Use `--api reference` for the matched manual configuration/ID path. The
automatic mode also checks its result against that manual path. Freeze the
compared package and map inputs with `--package-root` and `--assets-root` when
collecting before/after evidence. The numerical workload must run on the GPU.
Having a CPU backend available for
host setup does not make this a CPU speed comparison. Compare the committed
reference and the new API with identical configurations, actions and retained
outputs. Check optional full metrics and replay capture separately. Report the
actual workload and any missing evidence; do not replace a large batch with a
smaller one or treat a correctness pass as a speed result.

The following CPU/GPU command belongs to the **historical metric comparison**.
It documents how those records were produced; its older sizes are not approved
for a new GPU run under the current rule:

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m scripts.dev.benchmark_evaluation \
  --metrics-only --map-id 48 --sizes 64 128 256 512 1024 --repeats 5 \
  --output artifacts/m8-full-metric-performance
```

This runs each batch's ALPHA-versus-BETA games **once**: 1,984 episode executions
across five sizes. It then reuses the authoritative facts for the full numerical
CPU/GPU comparison, without rerunning games for warm metric measurements.
Compilation, first execution, warm measurements, host transfer, CSV writing and
memory are reported separately. No smaller batches are substituted.
Use a new output directory for each qualification. Keep the historical evidence
directories below intact. The current schema needs its own measured results;
the retained older results do not establish its speed or peak memory.

For broader rollout/capture work, omit `--metrics-only --map-id 48` and explicitly
pass `--sizes 32 64 128 512 1024`; do not inherit the older default size list.
That developer profile cycles five maps with exploratory policies and varied
rosters, and repeats complete GPU rollouts for none, priority, full, sparse full
and selected replay capture. It is intentionally more expensive. Optional
`--rollout-modes none full` narrows those repeated modes; without a measured
`none` reference, added rollout time is unavailable.

### Schema 5 column audit: 2026-09-10

This historical column audit ran 84 complete GPU games: all 42 training maps,
with ALPHA and BETA on both sides, canonical mirrored classes, and root seed zero. The
games covered 17,034 ticks, with lengths from 151 to 300. No replays or run
files were saved. The audit read full measurements in memory.

This sample did not vary the class order. Its 4,517 always-blank columns and
1,311 columns that were zero whenever defined were investigation clues, not a
removal list. Several classes may share a team, a class may be absent, and any
active slot may hold a different class. Separate checks of the game rules and
roster tests are needed before removing a column.

Schema 5 removed the 206 columns listed in the
[metric specification](metric_specification.md#schema-5-removals), keeping
11,114 numerical columns. Its full result was 1,030 logical bytes smaller per
lane than schema 4. This is an output-size calculation, not a measured speedup.
The running collector still needs its shared counts. No performance matrix was
repeated for this change; the measurements below describe their named older
schemas. They do not establish throughput or learning performance for schemas
5 through 13. Schema 6's 12 additions brought its logical full output to 55,630
bytes per lane. Schema 7 adds a further 100 output bytes per lane and 40 bytes
of running counts. These are storage calculations, not new performance
measurements; the schema-7 full output was 55,730 bytes per lane.

The before/after check compared all surviving values by column name over
30 cases and 1,631 replay boundaries. Every value and valid/blank flag matched
exactly, covering 18,126,934 cells. Cases included complete episodes, all five
classes moved through every slot, smaller and asymmetric teams, repeated
classes, initial statuses, shields and accepted Ultimates. These are focused
correctness checks, not an exhaustive game campaign. The raw results and source
hashes are in `artifacts/m8-schema-5/after-numeric-check.json`; the exact removal
and replacement list is in `artifacts/m8-schema-5/removals.json`.

### Schema 3 full metrics: RTX 5090, 2026-09-09

All five requested batches completed with **11,320 numerical columns** and
11,350 run-table columns including identity. The comparison uses the same
map-20, canonical mirrored 5v5, plain ALPHA/BETA workload and hardware as the
preserved schema-2 results below: RTX 5090, Ryzen 9 9950X3D, JAX 0.10.1,
CUDA 13, driver 580.173.02, with preallocation disabled.

Exactly **1,984 episodes** were executed, once each. These deterministic policies
again produced the same 172-transition match in every lane, with 128 padding
slots in each fixed 300-step cohort: 341,248 real transitions and 253,952 padding
slots overall. Every lane was simulated; the metric repeats reuse those facts.
This homogeneous comparison does not establish varied-policy or learner throughput.

The numerical measurements include initialization, collection over every real
and padded slot, and full finalization on resident facts. Five synchronized warm
executions give median (minimum–maximum) times. Integer results and validity
match CPU/GPU exactly; float32 results pass the unchanged `rtol=3e-5, atol=0.002`
comparison. Controlled effects now use float32 pairwise products and reductions.

| Environments | GPU full metrics, ms | CPU full metrics, ms | GPU ms/episode | CPU ms/episode |
| --- | --- | --- | --- | --- |
| 64 | 8.227 (7.957–8.382) | 36.709 (36.104–37.501) | 0.1286 | 0.5736 |
| 128 | 9.107 (8.914–9.148) | 61.226 (60.317–61.300) | 0.0712 | 0.4783 |
| 256 | 10.267 (9.801–10.583) | 101.475 (99.716–103.041) | 0.0401 | 0.3964 |
| 512 | 11.822 (11.429–12.038) | 167.586 (166.142–170.085) | 0.0231 | 0.3273 |
| 1,024 | 13.900 (13.707–14.018) | 262.406 (260.566–264.210) | 0.0136 | 0.2563 |

The 5.53-fold wider numerical output does not cause proportional computation.
At 1,024 environments the GPU median is 1.2% above the retained schema-2 median
(13.730 ms), with overlapping observed ranges. At smaller sizes the new medians
are lower; CPU medians are lower at all five sizes. These are comparisons with
preserved earlier measurements, not a simultaneous isolated attribution study.
The implementation shares pair statistics and groups finalization by scope and
measure instead of constructing a separate traced computation for every column.

Compilation includes tracing and lowering. First execution excludes compilation.
Simulation includes both policies, Core and benchmark fact retention; it is
executed once and is not a warmed rollout with full metrics or a learning update.

| Environments | GPU metric compile, s | GPU first metric execution, ms | CPU metric compile, s | CPU first metric execution, ms | Simulation compile, s | One simulation execution, s |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 3.343 | 14.595 | 2.149 | 38.198 | 4.135 | 14.078 |
| 128 | 3.396 | 14.841 | 2.176 | 65.854 | 4.188 | 14.153 |
| 256 | 3.498 | 15.698 | 2.366 | 102.280 | 4.558 | 14.176 |
| 512 | 3.214 | 16.735 | 2.244 | 177.219 | 4.502 | 14.291 |
| 1,024 | 3.118 | 19.161 | 2.228 | 265.599 | 4.303 | 15.541 |

Wider output has a measurable transfer and persistence cost. Transfer and buffered
CSV writing, including flush/fsync, are single measurements per batch outside
the numerical repeats. Full CSV sizes below use decimal MB and include their
header; directory sizes additionally include priority rows and run details.

| Environments | Metric transfer, ms | CSV write + fsync, ms | Full CSV, MB | Run directory, MB | Worker RAM peak, GiB | Sampled worker VRAM, MiB | JAX live-allocation peak, MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1.653 | 104.066 | 3.903 | 3.929 | 2.189 | Unavailable | 256.113 |
| 128 | 2.241 | 189.474 | 7.194 | 7.243 | 2.383 | 864 | 210.731 |
| 256 | 2.534 | 361.247 | 13.777 | 13.871 | 2.672 | 1,120 | 395.712 |
| 512 | 4.068 | 675.400 | 26.945 | 27.129 | 3.231 | 1,850 | 701.415 |
| 1,024 | 7.360 | 1,400.194 | 53.278 | 53.643 | 4.387 | 2,874 | 1,410.347 |

RAM is whole-worker maximum RSS. Process VRAM was sampled approximately once per
second; sampling started after the 64-environment worker finished, so that value
is unavailable. JAX's allocator peak is available for every batch and measures a
different boundary, excluding driver/context costs. Short process-memory peaks
may be missed. Benchmark workers retain facts, CPU copies and compiled programs;
these figures are not learner memory requirements. The wider full CSV takes
1.40 s to write at 1,024 episodes, versus 0.49 s for the preserved schema-2 suite.

Logical storage must be distinguished from these observed peaks. The running
collector occupies **13,578 bytes per environment**, versus 6,842 in schema 2.
A full scalar result occupies **56,600 bytes**, versus 10,240. At 1,024 lanes,
those are 13.26 MiB of counters and 55.27 MiB for one result batch. Retaining every
result for a 128-step training chunk needs 6.91 GiB; sparse full selection does
not shrink that fixed output shape. Priority-only collection avoids it.

Replay inspection additionally retains a scalar result at every frame for fast
seeking. One 172-transition replay needs 9,791,800 logical prefix bytes (9.34 MiB),
including its initial frame. Retaining such indexes for all 1,024 episodes would
need 9.34 GiB; this is a storage calculation, not an allocation made by the
benchmark. The streaming full-episode collector does not require that history.
The four independent replay audits prepared 190–234 frames in 3.79–8.25 seconds
including replay loading and compilation, retaining 10.26–12.63 MiB of prefix
arrays each. Those are single diagnostic readings, not warmed numerical timings;
replay preparation and host presentation must not be advertised as the 8–14 ms
batch reduction above. Subsequent seeking uses the prepared index.

The viewer omits structurally impossible rows before JSON serialization while
CSV preserves the fixed schema. On the same final 189-transition replay prefix,
this reduces the response from 11,320 rows / 9.739 MB to 6,250 rows / 5.262 MB.
Five serialization-only repetitions measured medians of 70.41 ms and 38.45 ms
respectively (ranges 68.32–137.18 and 38.15–44.92 ms). Every retained row is exactly
unchanged, including meaningful unavailable ratios. This measures serialization
of already prepared summaries, not numerical collection or browser rendering;
evidence is `artifacts/m8-schema-3-audit/summary-transport.json`.

The approved comparison does not rerun native none/full rollout overhead,
replay saving, tournament fitting or learner updates. Historical rollout evidence
below remains historical. Neither sample efficiency nor the one-day competence
claim follows from this measurement.

Evidence is in `artifacts/m8-schema-3-performance/batch-N.json` and
`sampled-gpu-memory.json`, with exact samples, trajectory identities, completion
counts, compiler memory, source hashes and output paths. All five rows use the
same unchanged sources. The sorted compact JSON hash of the production source
hash map is `d4fb6cbb9d57beb0ed5aafe3189125ef80538d597d4adea1df505a1c8da7d196`;
the benchmark driver SHA-256 is
`45c5354c014115e75e5db1f383d7965e32dcf242532400df1de4c1d410713632`.
Independent artifact review checks all 68 recorded source hashes, all CSV headers
and episode identities, and all 4,063,232 retained schema-2 cells by name. Missingness
agrees exactly; available values pass `rtol=1e-5, atol=1e-5`, with maximum absolute
difference 0.0004883. Recorded action/position/health trace hashes match the earlier
workload. CPU/GPU numerical agreement is asserted during the measured process;
the CSV artifact audit does not claim to independently recheck CPU arrays that
were not saved. See `artifacts/m8-schema-3-audit/independent-performance-audit.json`.
The final roster review subsequently refined only class-view routing in
`analysis.py` and `metric_catalog.py`. Collector, Core, writer and benchmark
driver hashes remain identical; numerical names/order are unchanged and the
roster CSVs match field-for-field except their analysis-source digest. Original
measurement hashes are preserved, with the later presentation-source hashes
recorded separately. This bridge uses hash/schema/result evidence and source
review; a complete pre-correction source snapshot was not retained.

### Schema 2 full metrics: RTX 5090, 2026-09-09

All five sizes completed using the installed **2,048 numerical measurements**
(2,078 run-table columns including identity). Hardware: RTX 5090, Ryzen 9 9950X3D,
JAX 0.10.1, CUDA 13, driver 580.173.02; JAX preallocation disabled.
The fixed map was `tdm_map_id_20_three_body_problem_test`, with canonical mirrored
5v5 and plain ALPHA/BETA, without exploratory actions.

Each lane was simulated, but these deterministic policies produced the same
172-transition match in every lane. Each fixed 300-step cohort therefore has
128 padded steps per lane. This is the requested homogeneous workload, not
varied-trajectory or training-throughput evidence. Earlier varied-trajectory
qualification and the semantic audits below supply separate correctness evidence.

Numerical timing includes metric initialization, every padded transition update
and full finalization on already-resident facts. It excludes simulation, loading,
compilation, host transfer and formatting. Integer results and validity agreed
exactly between CPU and GPU; float32 results passed `rtol=3e-5, atol=0.002`.
Controlled-effect matrix products use `HIGHEST` precision. Five synchronized warm
executions provide the median and observed min–max below.

| Environments | GPU full metrics, ms | CPU full metrics, ms | GPU ms/episode | CPU ms/episode |
| --- | --- | --- | --- | --- |
| 64 | 9.872 (9.536–10.335) | 38.439 (35.850–38.523) | 0.1543 | 0.6006 |
| 128 | 10.624 (10.315–10.839) | 71.160 (69.939–73.846) | 0.0830 | 0.5559 |
| 256 | 11.829 (11.635–11.877) | 120.456 (120.190–125.093) | 0.0462 | 0.4705 |
| 512 | 12.454 (11.848–12.932) | 206.584 (203.642–211.209) | 0.0243 | 0.4035 |
| 1,024 | 13.730 (13.260–14.447) | 302.638 (299.954–305.415) | 0.0134 | 0.2955 |

Compilation includes tracing, lowering and compilation of the measured function.
The first execution excludes compilation. Initial setup/reset remains separate.
The one simulation execution includes policies, Core and retained benchmark facts;
it is not a warmed native-full-metric rollout or a learning update.

| Environments | GPU metric compile, s | GPU first metric execution, ms | CPU metric compile, s | CPU first metric execution, ms | Simulation compile, s | One simulation execution, s |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 12.575 | 14.057 | 12.588 | 40.051 | 4.203 | 14.411 |
| 128 | 12.815 | 14.098 | 8.906 | 71.271 | 4.229 | 14.485 |
| 256 | 12.652 | 15.166 | 12.412 | 123.406 | 4.786 | 14.518 |
| 512 | 12.568 | 15.740 | 5.418 | 209.708 | 4.640 | 14.637 |
| 1,024 | 12.597 | 17.324 | 5.162 | 305.862 | 4.427 | 15.462 |

Every scheduled episode completed: 341,248 real transitions and 253,952 padded
transition slots in total. Transfer and durable CSV export are single measurements
per batch, outside the numerical repeats. CSV directory sizes include metadata
and priority rows as well as the self-contained full rows.

| Environments | Real / padded transitions | Metric transfer, ms | CSV write + fsync, ms | CSV directory, MB | Worker RAM peak, GiB | Sampled worker VRAM, MiB | JAX live-allocation peak, MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 11,008 / 8,192 | 2.171 | 51.130 | 0.980 | 3.572 | 866 | 256.113 |
| 128 | 22,016 / 16,384 | 2.112 | 79.217 | 1.879 | 3.091 | 866 | 211.248 |
| 256 | 44,032 / 32,768 | 1.887 | 142.613 | 3.678 | 3.792 | 1,122 | 395.712 |
| 512 | 88,064 / 65,536 | 2.366 | 256.613 | 7.278 | 3.519 | 1,852 | 701.415 |
| 1,024 | 176,128 / 131,072 | 3.088 | 489.487 | 14.476 | 4.583 | 2,876 | 1410.347 |

RAM is whole-worker maximum RSS; VRAM is sampled once per second with
`nvidia-smi`, so short peaks may be missed. JAX's allocator statistic measures a
different boundary and excludes CUDA context/driver overhead. These workers
retain benchmark-only histories and CPU copies, compiled programs and allocator
pools. They are **not learner memory requirements**. For example, retaining
128 steps of current observations at 1,024 environments alone uses 6.27 GiB,
before model parameters, gradients, optimizer state and activations.

The expanded native-rollout overhead and replay-publication timing campaign was
not repeated after the user narrowed qualification to this fixed-map metric
comparison. The retained schema-1 rollout results below are explicitly historical;
they do not substitute for a schema-2 overhead measurement. No learning update,
sample-efficiency or one-day competence claim follows from this benchmark.

Raw evidence is in `artifacts/m8-followup-performance/fixed-map/batch-N.json`, with
all samples, source hashes, actual trajectory hashes and compiler memory sizes.
All five rows used identical production sources from the preserved schema-2 candidate;
production hash-map digest:
`a7715a204f3ff768e147392e355f2690901ce319e292f10f1dd55357033d8539`.
Benchmark driver digest:
`bc04db999667551567e458021be76c4d683467a4274b31e929e9d2c545628b33`.
No measured source changed during execution. Independent review checked every
CSV row, episode identity, header, execution count and source hash.

### Historical schema-2 counter consolidation and replay-prefix evidence

Removing the redundant status-application accumulator saves **360 logical bytes
per full-enabled environment**, while preserving all 108 exported application
columns. The collector moves from 5,072 to 4,712 bytes for the original suite.
With the 660 new measurements it occupies **6,842 bytes**, a net increase of
1,770 bytes over the original collector. This is array storage, not allocator or
whole-process memory. The isolated consolidation did **not** demonstrate a
consistent runtime improvement:

| Environments | Original GPU, ms | Consolidated GPU, ms | Original CPU, ms | Consolidated CPU, ms |
| --- | --- | --- | --- | --- |
| 64 | 9.760 | 9.108 | 48.676 | 43.715 |
| 128 | 10.648 | 10.898 | 83.778 | 86.199 |
| 256 | 10.266 | 10.482 | 146.742 | 143.563 |

These are five-repeat warm medians for the original 1,388 measurements on
identical varied-map facts before/after consolidation. Full native rollout
medians were 13.95–14.19 seconds across these cohorts, also without a consistent
improvement. Every retained integer/validity cell and physical trajectory agreed;
79 floating cells at B64 differed by at most 0.000244141 HP, and all retained
cells at B128/B256 were bit-identical. The contended B256 attempt was excluded;
the accepted row was rerun with an isolated worker. Raw samples, source hashes
and the 621,824-cell comparison are in
`artifacts/m8-followup-performance/{baseline,consolidated}/` and
`variant-comparison.json`. Different workload/mode populations prevent a
controlled whole-process memory comparison. Do not compare these varied-map
rows to the expanded fixed-map rows as a controlled expansion speedup.

For four complete saved episodes (189–233 transitions), warmed CPU preparation
of **every prefix** took 39.00–52.08 ms originally, 44.98–50.64 ms after isolated
consolidation, and 50.04–54.40 ms with all 2,048 measurements. These are ranges of
per-replay medians over five repetitions. The expanded first cold analysis took
8.92 seconds including JAX compilation; loading each large replay JSON took
3.75–4.79 seconds separately. Final CSV formatting took about 1.1–1.2 ms.
The prefix values and validity arrays occupy 1.95–2.40 MB per expanded replay.
Seeking reads this prepared index rather than recalculating the whole history.

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m scripts.dev.benchmark_replay_analysis \
  --replays PATH_TO_REPLAY --repeats 5 \
  --output artifacts/m8-prefix-analysis
```

The corresponding evidence is in `artifacts/m8-followup-performance/` under
`baseline-prefixes`, `consolidated-prefixes` and `expanded-prefixes`. All
1,197,844 retained prefix positions were bit-identical after isolated
consolidation. The expanded collector preserved names/order/validity; eight
Burst totals/fractions changed only floating reduction order, with maximum
absolute difference 0.000122071. Independent raw-trajectory accounting checked
all **1,767,424 scalar/validity positions** across the 863 prefix boundaries,
including every new measure. Focused public-trajectory tests additionally cover
smaller/asymmetric/repeated-class rosters, initial dead/status states, simultaneous
healing/damage, support credit and partial resets. These expensive comparisons
belong to development qualification, never ordinary episode collection.

### Schema 1 baseline: RTX 5090, 2026-09-08

These retained measurements cover the original 1,388-column suite, before the
schema-2 additions. They are not substituted for the follow-up qualification.
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

For an uncommitted development review, use `scripts/dev/check_gpu.sh --allow-dirty`.
That runs the GPU correctness checks without staging or committing and does not
qualify a release. The schema-3 review passed 3,980 Python tests, Python static
checks, 506 frontend units, all 83 browser cases across eight profiles, and seven
GPU diagnostic cases. The two corrected browser profiles and final catalog
checks were rerun after their last changes; exact shard coverage is preserved.

Before making a commit, finish formatting, stage the complete intended candidate
and run `scripts/dev/check_before_commit.sh`. This runs both complete local gates
and verifies that the staged candidate stayed unchanged. It intentionally requires
a nonempty staged candidate; use the individual checks above on a clean checkout.
