# Replay Viewer

Use Replay Viewer to inspect a recorded game without changing it. Open a local
replay, a checked sample, or a scripted demonstration. The Viewer draws recorded
frames, plays their recorded events, exports a battlefield image, and computes
measurements from the captured facts. It cannot stage actions, run another
simulator turn, reset the game, or record to a chosen destination.

A normal workflow is:

1. Open a replay with the launcher below and use the printed URL.
2. Pause at a frame, choose Oracle View or an agent's POV, and inspect Latest
   Transition and Upcoming Transition.
3. Use Visual Filters to choose which recorded effects to draw.
4. Export PNG for that view, or open TDM Evaluation Metrics and download CSV
   for the current tick or the full captured game.
5. Use Exit Replay Viewer or stop its terminal process when finished.

Use [Combat Debugger](combat_debugger.md) for live play and map/scenario
editing. New recordings use Replay V3. Historical V1 and V2 files retain their
original meanings. Current policy rows store ally/enemy flags and a local self
index; the Viewer gets Team A/Team B labels from the recorded roster. Changing
the view or frame never changes observations, policy assignments, or saved
bytes. Current NoSharedObs actor exports use POV V2; historical POV V1 is still
readable.

## Select Exactly One Input

Run commands from the checkout with `uv` available. Each invocation must choose
exactly one replay, sample, scripted scenario, or list operation:

| Selector | Purpose |
| --- | --- |
| `--replay PATH` | Validate and open a local canonical-format replay bundle. |
| `--sample-replay NAME` | Verify and open one checked-in sample by stable launch name. |
| `--scenario NAME` | Run one registered demonstration in a separate process, then open its validated replay. |
| `--list-scenarios` | List default scripted demonstrations and exit. |
| `--list-sample-replays` | List checked-in sample names and descriptions and exit. |

Open a local artifact:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json
```

Choose the initial frame and audience without opening the browser automatically:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json \
  --frame-index 12 --view pov --pov-slot 5 --no-open
```

List and open a checked sample:

```bash
./scripts/dev/run_replay_viewer.sh --list-sample-replays
./scripts/dev/run_replay_viewer.sh \
  --sample-replay death-respawn-shield
```

List and record a scripted demonstration for viewing:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The public browser options are `--frame-index`, `--pov-slot`, `--view
oracle|pov`, `--ranges`/`--no-ranges`, `--port`, and `--no-open`. Ranges start
hidden; `--ranges` opts in without changing the nine default effects. `--seed`
applies only when recording a scripted scenario; its default is `0`. Browser
frame index defaults to `0`, view to `oracle`, and port to `0` (a free port
chosen by the operating system). `--pov-slot` selects an active global slot for
Agent POV. By default, it uses the lowest recorded focal-policy slot. If the
replay has no focal slot, supply an active `--pov-slot` to open Agent POV. List
operations reject unrelated options. `--static` has its own narrow matrix
described below. Write complete option names; abbreviations are rejected.

## Scripted-Scenario Isolation

`--list-scenarios` shows researcher scripted demonstrations and omits the manual
`arena_5v5` laboratory. `stacked_team_auras` follows `aura_crossfire` in the
default catalog and demonstrates simultaneous Basics under two same-team Mage
and two same-team Warrior emitters per team.

Developer visual-stress demonstrations are excluded by default:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios --include-stress
./scripts/dev/run_replay_viewer.sh \
  --scenario max_status_stack --include-stress
```

Use `--include-stress` only with `--list-scenarios` or `--scenario`. It lets you
discover or run developer stress demonstrations. It is not a browser view option
and cannot be used with local replays or checked samples.

The launcher records the scripted demonstration in a temporary child process
with `JAX_PLATFORMS=cpu`. The child executes the registered commands and
publishes one V3 replay containing the captured frames, facts and episode
provenance. No metric sidecar or full metric computation is required. The parent
opens those bytes through the public loader before starting the Replay Viewer.
The read-only viewer process does not import or run simulator control. Only the
separate demonstration process performs those scripted turns.

### Authoritative-Battlefield Visual Coverage Rule

Every battlefield visual mechanic needs executable checks in both a default
researcher demonstration and an opt-in stress demonstration. A label or
description is not enough. Tests must find the actual fact in validated
transition events or Scene V2 rows. Status mechanics must additionally prove
their stable catalog token through natural expiry.

Any change that adds a battlefield effect, durable badge, status, aura,
ability-family presentation, or lifecycle presentation must extend the regular
and stress trajectories—and the paired semantic coverage contract—in the same
change. Prefer extending an existing coherent trajectory; add a new scenario
only when doing so keeps the demonstration readable. Research-space panels and
controls outside the authoritative battlefield snapshot are not governed by this
scenario-pair rule.

## Checked Sample Replays

The checked V1 bundle under `examples/replays/v1/` contains exactly three
replay/metric pairs plus `manifest.json`:

| Launch name | Source scenario | Coverage focus |
| --- | --- | --- |
| `death-respawn-shield` | `death_respawn_cycle` | Lethal damage through the first post-shield interaction. |
| `recovery-status-lifecycle` | `recovery_refresh_cycle` | Recovery, rejection, refresh, break, reapply, and expiry. |
| `mirrored-five-class-ultimates` | `mirrored_ultimates` | Reciprocal demonstrations of every class Ultimate family. |

Every sample uses an 18×12 map. The manifest records stable names, source
scenarios, transition/frame counts, event-kind coverage, byte lengths, hashes,
and actual source/runtime provenance. These are deterministic, unofficial
display demonstrations. They do not qualify benchmark performance, evaluate a
policy, or certify the current source tree or host machine.

Verify the complete seven-file set through the public loaders:

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory examples/replays/v1
```

New generation writes manifest V2 and three self-contained Replay V3 files. It
does not write metric sidecars or run metric reducers. The Viewer computes
metrics later from the recorded facts. The checked V1 pairs stay unchanged and
keep their original strict verification. Both versions bound file reads, hold
one directory snapshot, verify hashes and source facts, and validate events and
captured facts. Current source provenance adds the explicit CodeRevision V2
marker; all recorded source facts must still match.

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py --generate
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory artifacts/visual-debugger-samples/v3
```

Generation defaults to `artifacts/visual-debugger-samples/v3`. Verification and
sample launch defaults still use `examples/replays/v1`. Current generation
writes four files instead of seven. Each replay stores ten local self indices
per frame and keeps the existing 58-column unit rows. Its semantic verification
is one pass over the recorded transitions, as for the historical samples. This
storage change and the skipped metric work do not establish a runtime speedup.

Generation is a maintainer task and refuses to overwrite a destination. Use a
new, absent directory only after scenario source, tests, and public
documentation are frozen; verify the complete set before an explicit reviewed
publication. Never hand-edit one member or delete the checked directory merely
to bypass the no-overwrite boundary.

## Transport and Exact-Frame Summaries

| Control | Behavior |
| --- | --- |
| **Start** / **End** | Seek to the first or final captured frame. |
| **−10** / **−1** / **+1** / **+10** | Issue one clamped absolute seek. |
| **Play** / **Pause** | Serialize playback with at most one replay request and one presentation in flight. |
| Frame slider | Preview the target tick locally without a request; commit one exact absolute seek when the value is committed. |
| Tick label | Show the authoritative current and final simulator ticks joined from the recorded timeline. |

Buttons, committed slider seeks, view/range changes, and reconnects show a
static summary of the exact selected frame. It shows enabled durable facts and
the events that produced the frame together, with no animation in progress.
Pressing Play restarts the displayed incoming transition at logical time zero
when one exists, waits for its scaled presentation to settle, and then requests
the next frame. Each next-frame response animates only the recorded transition
that produced that successor frame.

**Latest Transition** is the recorded incoming transition `T_(n-1)` that
produced the displayed frame `s_n`; it is absent at frame zero. **Upcoming
Transition** is the recorded `T_n` out of `s_n` in the same Submitted / Accepted
row grammar. Both panels remain global researcher evidence in Oracle View and
Agent POV: they show every configured-active actor in canonical roster order and
never stage, predict, or execute an action. Agent POV fog of war applies only
inside the battlefield snapshot and its choreography. The roster likewise
remains a global researcher control, partitioning active agents into **Visible**
and **Not Visible** groups so any agent's POV can be selected at the same replay
tick. Upcoming is absent only when the displayed frame has no recorded
successor.

The eight supported rates are exactly **0.25×, 0.50×, 0.75×, 1.00×, 1.25×,
1.50×, 1.75×, and 2.00×**. A rate scales the complete presentation clock,
including animation phases, waits, and the replay terminal hold. It never
changes simulator ticks or artifact contents.

### Document Keyboard Shortcuts and Exclusions

Unmodified Left, Right, and Space are document-level shortcuts for previous,
next, and play/pause. They do not require the battlefield or timeline to be
active. The viewer deliberately issues no replay command when:

- Shift, Control, Alt, or Meta is held;
- Space is an auto-repeat event;
- the event originates in or under a button, input, select, textarea, link,
  disclosure summary, dialog, editable region, or ARIA widget such as a
  slider, textbox, combobox, spinbutton, or menu item; or
- replay authority is absent, offline, hidden, or not yet installed.

Modified shortcuts, events from native controls, and absent, offline, hidden, or
uninstalled replay authority preserve ordinary browser and assistive-technology
behavior. When complete replay authority remains visibly installed, Space
auto-repeat and Left, Right, or Space during a temporary in-flight fence are
consumed without issuing another command. This prevents the document or an open
disclosure from scrolling while the retained replay scene is still active.

## Audiences and Recorded Authority

Oracle View exposes the full authorized battlefield presentation. Agent POV
applies the selected recipient's fog of war to that battlefield: NoSharedObs
shows the recipient's local view, while SharedObs shows the same-tick visual
union authorized by its team observations. Activating a visible agent in Replay
Agent POV switches the recipient at the current tick; it never advances or
mutates the artifact. Artifact identity, completion/processing evidence, PNG
export, and metric download remain capabilities of the researcher tool in every
visual POV.

Under [amendment
A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution),
support for both views keeps old and custom evidence readable. It does not make
both information modes eligible for official benchmark claims. NoSharedObs
material is historical, diagnostic, or custom evidence and is noncanonical for
new official claims. Drawing SharedObs data does not give a run official status.
Official evidence must separately prove the canonical actor projection and the
exact configured-active, same-team, off-diagonal availability matrix on every
recorded frame. The Replay Viewer continues to render both regimes and does not
mutate, migrate, or relabel their artifacts.

The SharedObs visual union follows [specification amendment
A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It combines only recorded rows from the same decision tick and permitted
same-team sensors. It does not recompute geometry, visibility, line of sight,
action masks, mechanics, or simulator state. It does not add teammate masks or
history, rewards, policy/critic state, transition facts, or hidden Oracle facts.
This display union is not a ready-made SharedObs learning tensor.

A separate server-authorized corpse overlay keeps locally visible dead bodies
consistent with Oracle View. Python admits a corpse only from the same decision
tick and only when an authorized living sensor has it within recorded
observation radius and static line of sight. The overlay is used for corpse
painting and inspection and may additionally admit only a death/respawn
presentation cue and that cue's owned endpoint. It never admits another event,
moves an ability route, or changes policy input, masks, targeting, actions,
simulator or recorded transition semantics, or the replay artifact.

### Technical Frame

Technical Frame shows the recorded Episode, Task Mode, technical Map name,
Observation Mode, Episode Limit, and root/episode-stream Seeds in both live and
replay views. These are shared researcher metadata; they do not enter policy
observations. Unknown historical seeds remain explicitly unknown.

The task header also shows the friendly map name and its recorded split, for
example `Three Body Problem (Test Map)`. Approved identity requires a declared
map ID or unchanged authored source plus matching current or retained historical
packaged geometry. A custom layout stays `Custom Map`; historical records retain
their recorded name, or show `Map name unavailable` when no name was recorded. Historical free-form
names never claim an approved map ID or split. Geometry alone never assigns an
old replay to a split.

The [approved 2026-09-16 map update](../design/specification_amendments.md#approved-map-publication-on-2026-09-16)
changes the map selection for new games. Publication and focused replay
compatibility checks passed. It keeps earlier packaged identities in one
immutable history resource: opening an old replay shows that game's
recorded layout and version, not the newest map with the same ID. Old replay
files and their recorded results are not rewritten. A36 lists the approved
clearance limits and exact map exceptions.

After installing the map update, restart DevClient, Replay Viewer and any
long-running Python process to refresh the cached map catalog. Opening an old
replay after restart still shows the old game. Earlier replay and speed results
remain evidence for the map versions they actually used.

Replay Oracle retains its five existing technical fields:

1. Artifact digest prefix
2. Frame
3. Simulator step
4. Incoming transition, omitted at frame zero
5. Ordinary movement distance scale

Alongside shared episode metadata, Agent POV receives Frame, Simulator step, and
its conditional authorized Incoming transition. It never receives the canonical
artifact digest or movement scale through this panel. Completion describes how
the captured game ended. Processing describes whether host output preparation
succeeded. They remain separate facts.

Replay files generated by `RunWriter` use
`<technical-map>__episode-<id>__seed-<root>__stream-<episode-seed>__a-<policy>__b-<policy>__<full-sha256>.marlbg-replay.json`.
Policy display labels are sanitized and shortened to keep the ASCII basename at
most 255 bytes; the technical map name and full replay digest are retained.
Unknown seeds are written as `unknown`, and unnamed layouts use a digest-based
custom/recorded-map token. Explicit researcher-selected save paths and stored
resume references are unchanged.

## Visual Filters

Visual Filters contains 19 browser-local controls plus Ranges. Initially enable
Ultimate Ability Effects, Spawn Shield, Basic Ability Effects, Regeneration
Effects, Death Effects, Resurrection Effects, Scrolling Battle Text, Respawn
Wave and Death Announcer. Ranges start off, so the initial count is `9 enabled`.
Visual Filters and Roster start open. **Default Configuration** restores these
nine effects with Ranges off. The complete filter inventory is:

1. Aura Fields
2. Aura Modifier Badges
3. Duration Status Badges
4. Spawn Shield
5. Target Selection Visuals
6. Basic Ability Effects
7. Ultimate Ability Effects
8. Regeneration Effects
9. Cooldown Effects
10. Status Application
11. Natural Status Expiry
12. Freezing Trap Break
13. Status Clear on Death
14. Death Effects
15. Respawn Wave
16. Resurrection Effects
17. Spawn-Shield Expiry
18. Scrolling Battle Text
19. Death Announcer

Death Announcer names the killing team and shows a compact list of victims using
numeric agent identities and classes. Hover or focus a victim to see its
complete **Kill Contributors** list. Useful same-tick Priest support shares
credit; healing that is all excess does not. Team A's kills appear in blue on
the left, Team B's in red on the right. Contributor details use the existing
accessible event tooltip; historical missing attribution is explicitly
unavailable.

The map label reads either the recording's map metadata or its verified authored
layout name, revision and digest. Older authored replays such as Three Body
Problem r3 therefore retain their recorded map identity. Geometry alone does not
assign an unnamed historical recording to a train, validation or test split.

Notices last 1.5 seconds at normal playback speed and remain readable on their
paused/static transition. Seeking, restarting or replacing the replay clears the
previous notices. These researcher HUD details are available in both POVs and do
not change policy observations or battlefield visibility.

Duration Status Badges includes the white crossed-swords **In Combat**
countdown. Basic Ability Effects and Ultimate Ability Effects each own their
corresponding activation presentation and damage/healing impact glyphs.
Scrolling Battle Text owns the complete net-health unit: outcome glyph, signed
value, recipient label, and connector. Those parts are enabled or disabled
together; damage and healing remain distinguished by their outcome sign and
color rather than by separate filters.

A filter change pauses playback and reinstalls the current settled summary after
filtering, so disabled paint never consumes layout space. Filters and Ranges do
not change authorized data or authorized event data used by battlefield
choreography. **Enable All** enables all 19 controls plus Ranges; **Disable
All** disables all 20 visible controls.

## PNG Export and Metrics

**Export PNG** is enabled only when a matched replay frame and presentation are
connected, visible, settled, and free of pending replay/presentation work. It
exports only the battlefield at twice its displayed pixel dimensions. The
toolbar, timeline, and inspectors are outside the image. The result uses the
bundled fonts and locked battlefield background, reflects the current audience,
selection, Ranges, and visual-filter states, and embeds one standard
`MARL-BattleGrounds Replay Provenance` iTXt record. Export does not navigate the
replay or request another replay frame.

The scoreboard derives the task, participant identities, current scores and
configured target from the captured episode. Victory, draw and defeat appear
only when the current frame contains the task completion event. Seeking backward
removes the later result. Legacy task-zero replays say **Combat diagnostic**.
Changing between Oracle and Agent POV preserves the cursor and playing intent.

Open **TDM Evaluation Metrics**, below Comprehensive Agent Class Details, to
prepare the offline analysis once. **Up to Current Tick** is the default: local
frame k uses exactly k captured transitions. The displayed tick comes from the
captured simulator state, which may start above zero. **Entire Episode** selects
the complete captured prefix; interrupted recordings do not become completed
results. An indeterminate progress indicator appears while metrics are prepared.
Longer replays can take longer; the indicator makes no fixed-duration promise.
Playback, seeking and POV changes remain available while analysis prepares. CSV
preparation also shows an indicator when the metrics panel is closed. The
**Topic** selector starts with **Episode Results**. It has 27 topics under six
headings, including five named Ultimates and **Team Formation**. Sixteen topics
have separate **Totals** and **By Recipient** views; eleven have one table.
**Respawning** shows each team's wave count, mean agents returned per wave, and
mean waiting ticks seen in the recording. **Deaths and Time Dead** keeps death
counts and dead time. Totals hold team-wide and acting-agent measurements. By
Recipient holds affected-agent totals and team-to-agent or agent-to-agent
details. Related views share no rows. Different topics can reuse an existing
measurement to help explain it; this does not add a CSV column or calculation.

Inside the five named Ultimate topics, rows and tooltips use **Mage Burst**,
**Warrior Charge**, **Hunter Trap**, **Rogue Poison** and **Priest Salvation**.
Other topics keep their shared measurement names. When a share uses a combined
team total, its tooltip still names all abilities included in that total. These
local wording changes do not change the numbers.

Schema 13 adds **Warrior Charge Applications**, **Rogue Poison Applications**
and **Priest Salvation Applications** for each team: six new columns. Their CSV
names are `team_{a,b}_warrior_charge_applications`,
`team_{a,b}_rogue_poison_applications` and
`team_{a,b}_priest_holy_word_salvation_applications`. These team rows appear
only in their named Ultimate topics. All existing CSV names remain unchanged,
including Slow, Stun and Anti-Heal application columns. Status Applications
keeps those separate effect rows. Mage Burst and Hunter Trap retain their
existing team application columns. Earlier CSV files stay untouched. See the
[schema-13 column
contract](../evaluation/metric_specification.md#schema-13-team-ability-application-columns).

**Find a Measurement** searches all 11,158 numerical columns. An exact CSV name
is checked first and returns that column, including one that does not apply to
the recorded roster. Other searches use measurement names, topics, related words
and recorded agent identities. The start of a word works too: `regen` finds
regeneration; `crippl pois` finds Crippling Poison. `Overhealing` finds excess
healing, and `spread out` finds teammate distances.

Search keeps who acts and who receives the effect separate. For example,
`healing from Priest to Mage` differs from `healing from Mage to Priest`.
`Damage received by Priest from Warrior` asks for the Warrior's damage to the
Priest. Add a team or an agent ID to make the identity more specific. A class
named inside an effect does not identify that effect's recorded caster: `damage
received while slowed by Rogue Poison` describes the damaged agent's status. It
does not say which Rogue caused the slow.

Search checks the kind of measurement and these roles before looking at related
words. A description cannot turn regeneration into damage or reverse the giver
and receiver. Applicable results come before inapplicable results; direct names
come before related detail. Search explains why an inapplicable column cannot
apply. Searching an ability name puts its own applications and effects before
shared context such as Total Kills. `Team A Charge` means Team A's Charge
actions; `Charge Slow Count` asks for the separate Slow application count. It
supports common measurement questions, not arbitrary English. Conditions such as
`damage without poison` or `damage after poison` show a clear explanation
instead of guessing a different measurement. Applicable matches open their table
and focus the named measure. A bright yellow outline and light tint mark the row
for three seconds. Choosing another result moves the cue; choosing the same
result starts its three seconds again. The cue disappears without moving
keyboard focus. It does not flash or move, including when reduced motion is
enabled. The search list is loaded once per replay analysis. Seeking, scope
changes and POV switches preserve it; replacing the replay clears it. Identity
fields remain in Episode Details and the column guide. Click outside the search
area to hide its results. Click or focus the search box to reopen the same list.
Your text and results loaded with Show More stay. Escape also hides the list
without clearing your text. Use Up and Down to highlight a result, then Enter to
open it. The cursor stays in the search box while you move through results, so
you can keep typing.

Scope, Topic, View, search, the definition and column headings remain outside
the scrolling rows. Rows show **Subject**, **Measure** and **Value**, with teams
first and numeric agent IDs using their recorded classes. Unavailable values
appear as a dash; hover or focus a measure for its definition, units and
missing-value rules. Formation summaries include mean distance and the number of
measurements taken while both allies were alive. Coordination and class
associations describe measured patterns. They do not prove that a policy learned
strategic reasoning.

**Download Metrics CSV** exports one wide row for the selected boundary, using
the same scalar names, order and values as the run tables; unavailable cells are
empty. Schema 13 exports 11,207 columns: 49 identity fields and 11,158 numerical
measurements. The export identifies its scope, local frame index, actual
simulator tick, and captured roster/policy identities. **Episode Details**
downloads the recorded episode, policy, completion and runtime metadata as JSON
without copying the trajectory or requiring a metrics sidecar. Older V1 replays
can be analyzed from captured facts without rewriting their artifacts. Derived
analysis records its own source fingerprint so it can be traced separately from
the original replay. Metrics remain researcher analysis in every visual POV and
never enter policy inputs, recipient tooltips or battlefield visibility.
Preparing analysis does not hold the replay command lock; subsequent seeks reuse
cached prefix components. Metric computation uses the captured game directly and
retains inexpensive consistency checks. Imported artifacts still receive the
input checks described below; computing metrics does not repeatedly revalidate
that same data.

## Static Matplotlib Frame

Render one exact frame without opening the browser or starting an HTTP server:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 12

./scripts/dev/run_replay_viewer.sh \
  --sample-replay recovery-status-lifecycle --static --frame-index 3

./scripts/dev/run_replay_viewer.sh \
  --scenario stacked_team_auras --static --frame-index 1
```

`--frame-index` is required with `--static`. Only the selected input, range
state, and—for a scripted scenario—seed/stress authorization are accepted in
this mode; browser audience, POV, port, and opening options are rejected. The
shell activates the optional `viz` dependency. Static mode validates the
complete bundle and paints the exact researcher frame through the stateless
scene-native Matplotlib adapter.

## Loopback and Artifact Safety

Local replays, checked samples, and newly recorded demonstrations are validated
as complete artifacts before the server opens a port or launches a browser.
Historical V1 companions are checked when present or required by the
checked-sample contract; current V3 replays are self-contained. The loader
rejects invalid schemas, noncanonical bytes, hash mismatches, broken event/frame
links, invalid frame indices or POV recipients, and unsupported paths. Its file
checks also reject prohibited symlink paths.

The browser server binds only to `127.0.0.1`, uses a random fragment-delivered
capability token, validates request headers and origins, serves an explicit
asset allowlist under restrictive Content Security Policy and `no-store`, and
handles replay commands one at a time with revision and command-ID checks.
Replay commands change only which recorded frame/view is selected; none can call
`step` or modify the source files.

Refresh and **Reconnect** fetch current authority without repeating a seek.
Closing the tab does not stop Python; use **Exit Replay Viewer** or `Ctrl-C`.

## Troubleshooting

- **No selector:** choose exactly one replay, sample, scenario, or list
  operation.
- **Artifact rejected before a URL appears:** inspect the reported path,
  canonicality, companion, frame, or POV error; the launcher intentionally did
  not bind a server.
- **Manual arena selected as a scenario:** open it with the
  [Combat Debugger](combat_debugger.md).
- **Stress scenario rejected:** add `--include-stress` to the scenario command.
- **Transport disabled:** reconnect if offline; Start/negative moves stop at the
  lower bound, positive moves/End stop at the captured endpoint, and artifact
  actions wait for a settled frame.
- **Metrics unavailable:** inspect the analysis panel's error. A missing V1
  metrics sidecar does not prevent analysis from recorded facts or the Episode
  Details download; current V3 recordings do not need a sidecar.
- **PNG export disabled:** pause playback and wait for the exact-frame summary
  to settle in a visible connected tab.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

Contributor prose follows the [Documentation
Standard](documentation_standard.md). The replay format itself is documented in
[replay_format.md](../evaluation/replay_format.md). Return to the [browser-tools
migration page](visual_debugger.md) or the [project README](../../README.md).

Current DevClient and scripted-scenario recording captures the replay without
computing the full metric suite. Opening Evaluation Metrics or requesting CSV
opts into full replay analysis once; the result is cached across cursor and POV
changes. Library callers likewise opt in with `analyze_replay(bundle,
full=True)`. Research training and evaluation independently select priority/full
metrics and replay saving through the [evaluation
workflow](../evaluation/workflows.md).
