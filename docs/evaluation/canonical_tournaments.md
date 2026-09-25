# Canonical Tournaments

`run_canonical_tournament` compares the twelve controller versions in one released
snapshot. Supply one System, Policy or built-in name to add one challenger.
The twelve alone have 66 matchups. Adding one challenger adds 12 matchups.
Every matchup keeps its Team A/B assignments and exchanges both complete spawn
banks between paired games. The challenger is Team A in its games.

The tools do not invent an official Big 12. A released snapshot needs separately
approved rules, qualified controllers and complete supporting records. Until that
bundle is installed, a default canonical call gives a missing-bundle error.
Local execution never admits a challenger or changes the published twelve.

## Use The Package Commands

The command routes call these same Python authorities. Once a qualified snapshot
and the needed assets are available:

```bash
python -m marl_battlegrounds canonical
python -m marl_battlegrounds canonical --system research_methods:load_selected \
  --output-dir runs/challenge
python -m marl_battlegrounds canonical --config released-snapshot.json \
  --rerun-existing --output-dir runs/fresh
python -m marl_battlegrounds tournament --config custom-tournament.json
```

The paths above are your selected configuration files; create or prepare them
before calling the command. The runnable fixture example below creates real
local paths. It never supplies an official default. A custom command obtains its
population and scientific settings from its config. Canonical budgets can use
`--games-per-opponent N`; a different budget is a research override, not admission
approval. `--rerun-existing` runs the whole resolved field, with no hidden fallback.

Resume with the exact printed saved run path using `--resume-from PATH`. Omit
scientific options to inherit saved conditions. In particular, omitting `--system`
on resume preserves the saved challenger; it does not remove it. Explicit
conflicts fail before recovery changes files. Already complete results are read
without another fit. Unsaved commands print bounded rankings before exiting;
use `--output-dir` for later table or replay access.

Inspect assets before preparing them:

```bash
python -m marl_battlegrounds models download --config released-snapshot.json \
  --roles outcomes_priority --dry-run
python -m marl_battlegrounds models download --config released-snapshot.json \
  --roles outcomes_priority --cache-dir prepared-assets \
  --output-config prepared-priority.json
python -m marl_battlegrounds canonical --config prepared-priority.json
```

Inspection creates no files and makes no network request. The download route
prints missing bytes and asks once; `--yes` deliberately confirms noninteractive
use. `models download` alone selects model payloads, not reports or replays.
Priority reuse needs outcomes/priority records; request `full_report` separately
for `--metrics full`, and `replay` for selected replay coverage. Required metadata
follows the existing dependency rules. Tournament execution itself stays offline.

A custom cache requires `--output-config` so the next command can find its verified
files. The prepared config keeps the same scientific identity and holds absolute
path hints. Its parent directory must exist; a different existing output file is
never overwritten. The command prints the next needed preparation or execution
command. A prepared model does not imply available reports.

Cleanup is explicit: `models clean --sha256 DIGEST --dry-run` previews selected
content files; remove `--dry-run` to confirm deletion, or use `--yes`. Stop other
writers or downloads using those entries first. One digest may serve several
snapshots and saved runs. The command cannot prove an entry is unused and never
automatically prunes old versions, source bundles or files outside the chosen cache.

## Try The Workflow With Artificial Records

After installing MARL-BGs, run the complete
[researcher example](../../examples/canonical_tournament.py):

```bash
python examples/canonical_tournament.py fixture --directory runs/fixture --save
python examples/canonical_tournament.py fixture --entrants 13 --metrics none
```

The example creates its own fixture paths using
[one shared fixture builder](../../examples/canonical_fixture.py). It invents
outcomes and measurements and labels them as artificial. It uses the custom
configuration route. It does not install those versions as official controllers.
Without `--save`, it creates no result directory. The explicit fixture command
still writes its source fixture files; ordinary calls on existing prepared assets
create no files unless an output directory is supplied.

The example checks saved-first resume and prints the actual saved directory.
Read that exact directory later:

```bash
python examples/canonical_tournament.py read YOUR_SAVED_RUN --table matches
```

Reading uses bounded table batches. It starts no simulation runtime, downloads
nothing and performs no rating fit, recovery or file repair.

## Compare A Released Snapshot

Once a real release and its assets are installed:

```python
import marl_battlegrounds as marl_bgs

incumbents = marl_bgs.run_canonical_tournament()
print(incumbents.reused_games, incumbents.executed_games)
print(incumbents.table("tournament_rankings"))

challenged = marl_bgs.run_canonical_tournament(
    "random", metrics="priority", output_dir="runs/challenge"
)
print(challenged.table("tournament_headline_metrics"))
```

The built-in name demonstrates the call. It is not a claim of eligibility.
A provable exact incumbent duplicate is rejected even under a new display name.
Use `system=None` for the twelve alone. Distinct methods also need distinct display
labels. Unknown provider or closure behavior stays unknown; matching incomplete
metadata does not prove two controllers are the same.

`config=None` selects the installed released snapshot once. A JSON path or parsed
mapping can select the same unchanged release explicitly. Moving identical asset
files does not change its scientific identity. An edited population or protocol
belongs in `run_tournament(config=...)`. You may also continue to call
`run_tournament([system_a, system_b, ...])` directly. Custom fields can be larger
than thirteen. Supply one population source, not both.

A configuration owns its scientific settings. Do not separately supply maps,
seeds, rosters, weights or stopping rules on the configuration route. On the
canonical route, `games_per_opponent` is the explicit budget override. Omitted
values inherit the snapshot budget. An explicit equal value is still compatible.
A different positive integer is a research override and cannot qualify promotion.
Five-map coverage requires a multiple of ten. The same budget applies to every
matchup, including reused games. The library selects balanced complete pairs in
the stored order, without looking at outcomes.

`rerun_existing=True` runs the entire resolved tournament. It does not fill only
missing games, and a failed reuse check never chooses it automatically. Fresh
execution needs its models, exact conditions and schedule evidence. It does not
need the old measurements it will replace.

### Snapshots And The Red Zone Rule

Team Deathmatch gives the enemy team 2 points instead of 1 when an agent dies
inside its own team's Red Zone ([A44](../design/specification_amendments.md#a44-team-deathmatch-red-zone-scoring)).
A snapshot keeps the rule its games were played under. The depth sits inside
each source configuration asset, so the source configuration ID and the
snapshot ID already cover it. The snapshot has no separate depth setting.
`run_canonical_tournament` takes no `red_zone_depth`, and
`run_tournament(config=..., red_zone_depth=...)` fails before any write, because
the configuration owns its rules.

A snapshot saved before the rule pins scalar schema 14, run schema 2 and
replay schema 3. Its 12-key configurations read as depth 0.0 (one point per
death) under their original IDs, so the snapshot keeps its identity. Its
recorded games can still be read and reused. Any call that needs a new game,
such as adding a challenger or `rerun_existing=True`, stops before writing
anything: "Snapshot configurations were saved before the Red Zone rule;
recorded games can be reused, but new games need a snapshot prepared with
current configurations." Prepare a new snapshot from current configurations to
play new games under the rule. Editing a source configuration is refused even
when its file hash and snapshot ID are recomputed: the edited content no longer
matches its recorded configuration ID.

## Prepare Assets Explicitly

Tournament execution is offline. It verifies required hashes, sizes and recorded
coverage before starting new games. It loads models only for active matchups.
Complete twelve-only reuse makes no model loads or action calls.

Use the one asset helper to inspect missing files or explicitly download them:

```python
from pathlib import Path
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.tournament_assets import prepare_tournament_assets

inspection = prepare_tournament_assets(roles=("outcomes_priority", "full_report"))
print(inspection["missing"], inspection["bytes_missing"])

prepared = prepare_tournament_assets(
    cache_dir=Path("prepared-assets"),
    roles=("outcomes_priority", "full_report"),
    download=True,
)
result = marl_bgs.run_canonical_tournament(config=prepared["config"], metrics="full")
```

This example requires a released snapshot. `download=True` is the explicit network
request. Report-only roles do not request model payloads or replays. Required
configuration, schedule, registration and compatibility metadata accompanies the
selected payloads. Pass the returned config to use a chosen cache directory;
verified path hints do not alter the snapshot identity. On resume, prepare the
saved snapshot first. An explicit different snapshot fails before preparation.
The example chooses only the payload roles needed for its requested work.

Models are inference assets. Training checkpoints, optimizer state and opaque
provider sessions are separate. A trusted installed factory or loader returns
the method. Its actual parameter/template evidence must agree before it acts.
Hashing downloaded bytes establishes integrity, not the safety of arbitrary code
or scientific qualification. No Python code is downloaded by the helper.

## Read Results And Resume

The result uses the ordinary interface:

```python
import marl_battlegrounds as marl_bgs

result = marl_bgs.run_canonical_tournament(output_dir="runs/big12")
assert result.run_dir is not None
loaded = marl_bgs.load_results(result.run_dir, phase="tournament")
for batch in loaded.iter_table("matches", rows=32):
    print(batch["episode_id"], batch["outcome"])
resumed = marl_bgs.run_canonical_tournament(resume_from=result.run_dir)
```

Omitted scientific and capture choices inherit the saved run before installed
defaults are consulted. Explicit conflicting choices fail before recovery changes
files. Worker count and chunk size can change. A saved raw challenger without a
loadable immutable descriptor must be supplied again with matching identity.
Missing new games resume with their original keys and IDs. A complete saved
summary loads without fitting again. Summary-only finalization needs no actions.

`CanonicalTournamentResult` adds `snapshot_id`, `challenger_id`, `planned_games`,
`reused_games`, `executed_games`, `games_per_opponent` and `protocol_compliant`.
Counts cover the whole logical run, including earlier successful attempts.
`metadata["executed_this_call"]` counts only this call. `big_12_id` aliases the
snapshot ID; `protocol_id` names its immutable rules. Protocol compatibility is
not admission approval.

Raw rows retain their original run, phase, pass, episode and Team A/B owners.
Logical IDs select captures; the library maps them to original IDs and keys.
Researchers do not merge source rows or rebuild ownership. Run-wide, tournament-
phase and coordinator views cover the whole selected population. An execution
pass covers its local games. Repeated episode IDs in different origins remain
different games.

Reused wide reports stay in their verified source files. A saved result records
references rather than duplicating those reports. Retain the referenced assets
when moving or archiving the result. Supply an unchanged config with updated
verified file paths when resuming from a new cache. The run saves these location
hints while keeping its original config bytes and scientific identity. No-file
results also depend on those assets
for later lazy reads. A missing required source raises an error for the dependent
table; it never produces a silently incomplete table. Stored summaries remain
readable without unrelated model or full-report assets.

The default is `metrics="priority"`. Full data can satisfy priority requirements.
`metrics="none"` keeps outcomes and rankings but skips headline calculation and
optional measurement parsing. Selected full reports and replays remain separate
requests; their presence does not upgrade the whole run. Select them with logical
`full_metrics_episodes` and `replay_episodes`, or use `save_replays=N` for the first
N scheduled games. Conflicting selections fail. Missing reused replay/full
coverage requires preparation or an explicit fresh run.

Ratings are fitted jointly from all qualifying game records. Old Elo supplies
no prior credit. The existing estimator, weighting and bootstrap rules remain
unchanged. The shared ranking view uses unrounded stored Elo; no duplicate ranking
CSV is written. Headlines use the same verified participant ownership and exact
measurement rules as ordinary tournaments.

## Maintainer Admission Is Separate

The private maintainer helpers live in `evaluation.admission`. They manage an
explicit local store, with an unpublished provisional twelve separate from the
published snapshot. Researcher calls never touch that store. A real store must
start from an approved pinned snapshot. Creating and qualifying the first real
population is a separate approval step. Only isolated fixture stores may start
from artificial custom records.

Admission uses complete submissions in their saved order, approved eligibility
and resource rules, full metrics from the first challenger transition, and one
joint thirteen-entry fit. The challenger must be strictly above the lowest
incumbent. An exact cutoff tie does not promote. An unapproved choice among tied
lowest incumbents blocks the decision. A promotion retains 55 incumbent matchups
and 11 challenger matchups, then fits those twelve afresh. Retries keep the saved
schedule and keys and cannot apply membership twice.

A store may start from a snapshot saved before the Red Zone rule; it keeps its
original snapshot ID. A challenger's games would be new games under the current
rule, so executing that admission stops with the message above before any run
directory or store change.

The release instant is 00:00 Europe/London on the first day of the month. Its
cutoff is 72 elapsed hours earlier. The store records both local and UTC instants.
Late submissions and unfinished attempts roll forward; they cannot change a
frozen edition. Preparing a release does not publish it. Publishing is a separate
explicit maintainer action. No package release, network push or website deployment
happens implicitly.

Official budget, resource/timing definitions, affected scientific eligibility,
revision/adaptation rules and tied-incumbent eviction need their own approvals.
Tests and artificial examples qualify machinery only. They cannot provide those
approvals or substitute for a released, scientifically qualified Big 12.

## Measured Costs And Limits

The Packet 7 check on 17 September 2026 used an RTX 5090, batch 32 and chunk
size 16. Three small controllers played 192 games, each lasting four transitions.
That is 768 real transitions, plus 2,304 padded lane rounds. The numbers below
are medians of five complete warm calls. They include setup and result handling;
they are not the time for one game or one replay.

| Work | Time | Real Transitions Per Second |
| --- | ---: | ---: |
| Canonical orchestration without saving | 3.14 seconds | 244 |
| Saving outcomes, two full reports and two replays | 5.77 seconds | 133 |

The matching direct evaluator took 3.20 seconds; the sample ranges overlap.
The unchanged committed path took 3.09 seconds. These comparisons show no clear
added no-save cost for this small workload. They do not prove a speedup. Saving
is a substantial extra cost when games are this short. Its output was 1.90 MB.
About 0.50 seconds of the no-save call was spent in synchronized evaluator chunks;
the rest includes game setup, completion handling and host work.

The first calls took 10.69 and 24.23 seconds. Observed backend compile/load events
accounted for 5.54 and 14.52 seconds respectively; this is not a complete isolated
compiler measurement. Warm calls and a same-shaped weight change added no such
events. At most two loaded controller models were live at once. Peak live device
allocation was about 206–208 MB, while JAX's reserved memory pool was about
25.2 GB. Process peak RAM was 2.41–2.83 GB, including compilation and setup.
These tiny controllers do not prove that arbitrary large models fit together.

Explicit `device_get` calls carried about 0.36 MB without saving and 10.87 MB
with the selected captures. These are logical payload counts, not measured bus
traffic; implicit transfers are excluded. Reusing 660 artificial twelve-entry
records took 0.67 seconds with outcomes only and 1.91 seconds with full reports.
Both made zero model loads and zero action calls. Full reports added about
1.24 seconds to verify and read 660 wide rows, about 15.5 MB of source data.
About 0.50 seconds remained in exact configuration preparation. This is a
visible setup cost: saved declarations still pass the shared physical validator
before they can support a new comparison. It is not a per-transition cost.
The fixture outcomes use a degenerate bootstrap case, so they do not measure
an expensive nondegenerate 5,000-replicate rating fit. The estimator settings
were unchanged. No learning, official eligibility or theoretical speed limit
is established by these checks.

The opt-in `scripts/dev/benchmark_evaluation.py --canonical-contracts` route
accepts one declared workload and one case (`shared`, `canonical`, `capture` or
`reuse`). It preserves normal benchmark defaults. Retain its workload, source
hashes and raw JSON when making a comparison.
