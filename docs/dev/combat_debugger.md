# DevClient and Combat Debugger

Use the DevClient to play and inspect a live game, edit maps, and build Team
Deathmatch scenarios. Its three areas are **Combat Debugger**, **Maps**, and
**Scenarios**. Python runs the simulator and validates saved content. The
browser collects your input and draws the result.

A normal workflow is:

1. Launch the DevClient and use the default arena, or open Maps/Scenarios to
   create and save a starting state.
2. Load the saved revision in Combat Debugger.
3. Choose the controllers and information mode. Stage manual actions, then
   Submit one simultaneous turn for all active agents.
4. Inspect the result. Reset returns to the exact loaded starting state.
5. If recording was enabled at launch, use Finish & Review to save and inspect
   the recorded game in the [Replay Viewer](replay_viewer.md).

The task selector currently offers TDM. The scoreboard shows both teams'
controllers, scores, and configured score target. Scores are points: a death
gives the enemy team 1 point, or 2 when the agent dies inside its own team's Red
Zone ([A44](../design/specification_amendments.md#a44-team-deathmatch-red-zone-scoring)),
so a score can differ from the kill count. A completed game also shows
Victory, Draw, or Defeat. Older neutral setups appear as **Combat diagnostic**.

Live recording captures the game without computing the full metric suite. Replay
Viewer computes those measurements when you open analysis or request CSV.
Finishing a recording shows a progress indicator while it prepares and saves the
replay; it does not promise a fixed wait. Checked samples, scripted
demonstrations, and existing replay files also use Replay Viewer.

## Launch

Run these commands from the repository checkout. The launcher uses `uv` and the
project Python environment. Open the default Oracle View:

```bash
./scripts/dev/run_dev_client.sh
```

Useful launch variants:

```bash
# Start with one active global slot selected and hide ranges.
./scripts/dev/run_dev_client.sh --controlled-slot 5 --no-ranges

# Start in the selected actor's authorized POV.
./scripts/dev/run_dev_client.sh --view pov --controlled-slot 5

# Print the loopback URL without asking the operating system to open it.
./scripts/dev/run_dev_client.sh --no-open --port 8123

# Show the executable CLI contract.
./scripts/dev/run_dev_client.sh --help
```

The launcher finds the repository from its own location and listens only on
`127.0.0.1`. By default, the operating system chooses a free port. The launcher
prints the URL and tries to open a modern browser. Node.js and npm are needed
for frontend development, not for using the installed browser assets.

`run_debug_renderer.sh` remains a thin compatibility redirect to this launcher.
The Python entry point also sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before
loading the simulator, unless the caller explicitly set it. JAX then allocates
GPU memory as needed; this is not a memory limit. Calling `main()` after JAX has
initialized cannot release an existing allocation pool, and prints a warning.
Set the variable before starting Python to use this startup default reliably.
Help and argument checks do not initialize JAX or change memory settings.

The public options are:

| Option | Meaning |
| --- | --- |
| `--record-replay PATH` | Record one live episode to a self-contained V4 replay, then offer read-only review. |
| `--seed N` | Set the deterministic reset/step seed; default `0`. |
| `--controlled-slot N` | Select an initially active global slot; otherwise use the arena default. |
| `--static` | Render one stateless Matplotlib reset snapshot without a browser server. |
| `--no-open` | Print the URL without opening a browser automatically. |
| `--port N` | Select a loopback port; `0` requests an ephemeral port. |
| `--view oracle\|pov` | Select the initial authorization; default `oracle`. |
| `--ranges` / `--no-ranges` | Show or hide controlled-actor ranges initially; default hidden. |
| `--execution-information-mode shared_obs\|no_shared_obs` | Select policy information at launch; default `shared_obs`. Reactive controllers require SharedObs. |

Write full option names; abbreviations are rejected. Replay, sample,
scripted-scenario, frame-index, and replay-POV-slot options are rejected with
the Replay Viewer launcher named in the error.

## Maps and Scenarios

Maps and Scenarios each show an object or roster list, a map, numeric fields,
and Python validation problems linked to those fields. Opening either area
creates an untitled draft without a prompt. Choose a lowercase snake-case asset
ID, such as `training_map_10`, on the first Save. Later Saves create a revision
of that same asset only if the saved revision has not changed elsewhere. The
visible Name field remains free-form. Drag centers for quick placement, hold Alt
to bypass the fixed 0.5-world-unit snap, use arrow keys for exact nudging, and
use the inspector for dimensions, wall size/rotation, roster, episode, current
state, name, description, and ordinary notes. Keep experiment hypotheses,
treatment roles, seed schedules, and measurements in the experiment definition.
The saved scenario describes the physical starting state; the separate official
evaluation-definition workflow remains future work. The mouse wheel zooms only
the authoring canvas; Space-drag or middle-button drag pans it. **Recenter**
restores the complete-map view without changing the draft. **Reset (R)**
restores the latest New, Open, or successfully saved baseline and can be undone
immediately. The `R` shortcut does not fire while typing in a form control. The
ten spawn-pad identities stay fixed. New walls and pillars receive `obstacle_0`,
`obstacle_1`, and so on. You can edit obstacle IDs using the Object ID field.
Deleting an obstacle does not rename the others; you can close a numbering gap
yourself. Obstacles also have duplicate, delete, and up/down order controls.

Drafts are local files under ignored `artifacts/dev_client/` storage. Each
selector lists the latest revision of every applicable saved asset, with
numeric-aware ID order (`map_9` before `map_10`). Use ordinary browser scrolling
and keyboard navigation. The status line names the exact saved revision and
repository-relative path, for example
`artifacts/dev_client/drafts/maps/training_arena/r2.json`, and reports unsaved
changes while keeping the last saved location visible. Save is explicit; there
is no autosave. A successful Save persists through DevClient shutdown and
restart and refreshes every applicable selector immediately. Save checks the
exact revision, so an older browser cannot overwrite newer work. **Delete
Saved** removes the selected map or scenario identity and all of its saved
revisions only after an explicit confirmation. A stale revision cannot delete
newer work. If the deleted asset is open, its current browser content remains as
an unsaved copy that may be saved again; deletion does not interrupt an already
loaded Combat snapshot.

A new scenario can start blank, copy a saved map, or duplicate a saved scenario.
The adjacent source selector carries the exact saved revision, and successful
Save, Save As, and Delete refresh discovery immediately. Copied map content is
independent: later changes never propagate in either direction. The browser
edits JSON-shaped fields only. Python compiles the whole draft into existing
`EnvConfig` and `EnvState` authorities and runs the existing validators before
any scenario can enter the Combat Debugger. Save is the only way the DevClient
persists asset content, and Delete Saved is its only removal operation.
Experiment and evaluation manifests own any later approval, partition, or
normalized scientific identity.

Scenario drafts declare the Team Deathmatch Red Zone rule in the **TDM
episode** group. **Red Zone Depth** says how far each team's Red Zone reaches in
from its own spawn edge, in map units. When an agent dies inside its own team's
Red Zone, the enemy team gets 2 points instead of 1
([A44](../design/specification_amendments.md#a44-team-deathmatch-red-zone-scoring));
0 turns the rule off. A blank scenario and a copy of a saved map start at 5.0,
the default, and Duplicate keeps the source scenario's depth. Python checks the
value exactly as typed: a negative value (including -0), a positive value too
small for float32, or a value wider than the map is a problem linked to the
field. Scenario drafts saved before the rule (`dev-scenario-draft@1`) keep one
point per death: they read as depth 0 and keep their semantic digests. Opening
one for editing shows it at depth 0; the next Save writes the new draft version.
The Scenario Author canvas tints each team's Red Zone floor from Python's
latest validation of the current draft, so it shows no tint while a check is
pending, for map drafts, or at depth 0.

## Loading Saved Scenarios and Map Previews

The Combat Debugger selector lists every latest execution-valid saved map and
scenario revision in numeric-aware asset-ID order. Scenario rows load the
authored starting state. Map rows are explicitly labelled as default 5v5 TDM
previews: Python copies the map into the default scenario, with Red Zone depth
5.0 (`default-tdm-map-preview@2`), builds the simulator inputs, and validates
them. A map narrower than 5.0 map units instead gets a preview explicitly
labelled **Red Zone Off**, at depth 0.0 (`red-zone-off-tdm-map-preview@1`).
This preview choice never changes an authored scenario's declared depth. A
playable scenario whose depth does not fit its map still fails validation.
Neither preview changes or saves the map. `Open in Debug`
in either authoring area calls this same loading service for its current buffer.
When a map retains an approved TDM map's source name and content digest, its
preview is recorded under that map's registered identity: the map ID, name and
split appear in the recording keys, and the layout identity carries the approved
source's asset ID, catalog revision and semantic digest, so the match summary
shows the map's name instead of Custom Map. Any other map is recorded as a custom
layout.
Python parses and validates each requested start before replacing the current
session. A rejected load leaves the session untouched and reports linked problems.
Combat also shows the error in the page notice. If the connection is lost, the
notice says the load outcome is unknown; the client does not retry it automatically.
A successful later load clears the old failure notice.
Reset restores the immutable loaded snapshot and seed, including its map,
roster, scores, timers, and current timestep.

Team A and Team B offer **Manual**, **Reactive TDM ALPHA**, **Reactive TDM
BETA**, **Reactive TDM GAMMA** and **Random**. The simulator step and SharedObs
policy teams run compiled. The first Submit with a controller pair not yet used
in this DevClient process compiles for a few seconds; later Submits take
milliseconds. NoSharedObs Random runs uncompiled.
ALPHA is a deterministic SharedObs policy for all five classes. It chooses
movement and combat separately. Random samples only the exact current valid
action support and ignores observation features; the same key and mask therefore
produce the same action under SharedObs and NoSharedObs. Neither controller is
an official baseline or Big 12 entrant. There is no policy registry or
checkpoint loader. Under [amendment
A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution),
SharedObs is the default and the only evaluation-eligible information regime.
NoSharedObs remains available when both teams use Manual or Random, and in
generic custom research. It is disabled while either team uses a reactive
controller; the application never silently substitutes a regime or controller.

**Reactive TDM BETA** is a deterministic SharedObs-only
specialist. Mage, Warrior, Hunter and Priest use ALPHA unchanged. Rogue pursues
an observed living enemy Priest first, otherwise Mage, otherwise Hunter,
choosing lowest current HP within the selected class and then lowest global
slot. If none is observed alive, Rogue uses ordinary ALPHA movement toward the
nearest observed enemy or, without enemies, map center.

BETA allows clear paths and glancing shoulder contact at least 45 degrees from
head-on at the first contact with each observed living body in its path. It
ignores its own body and the selected prey in this check. Tiny initial overlap
within the geometry tolerance counts as contact. With deeper overlap, it permits
only outward moves that do not deepen the overlap. Away from wall steering, it
chooses a useful legal endpoint nearest the prey, then uses action order to
break ties. A clear route gets no extra bonus over allowed contact. Temporary
retreat is allowed; if no move qualifies, it chooses Stay. Ordinary simulator
collision response resolves the attempted contact. Scenarios 3 and 5 use
`scenario_5` behavior v5, displayed as BETA. The old standalone Scenario 3
controller was removed; it is not a BETA alias. Historical recordings retain
their original identities.

BETA chooses combat separately: use Ultimate on the lowest-health legal enemy
within Basic range; otherwise use Basic on the lowest-health legal enemy there;
otherwise choose no combat. The selected prey determines the destination, not
necessarily the attack target. BETA has no target memory, hidden prey positions,
or opponent-action predictions. It cannot guarantee a route. Dead or inactive
agents submit no-op actions, and the current action masks take priority over
controller preferences.

**Reactive TDM GAMMA** (`tdm_gamma`; the same controller as the Python name
`tdm-gamma`) is a deterministic SharedObs-only controller defined in [amendment
A43](../design/specification_amendments.md#a43-reactive-tdm-gamma). Version 2
plays exactly like BETA except for three rules:

- **Search.** With no living enemy in view, Warriors, Mages, Hunters and Rogues
  walk toward the middle of the enemy team's five spawn pads, and stay once
  they are there. Priests keep BETA's rules.
- **Trap hold.** Warriors, Mages, Hunters and Rogues never damage an enemy with
  2 or more Hunter Trap ticks left. This covers every Basic, Warrior Charge and
  Rogue Ultimate, and a Mage Bursts only when it has a legal Basic target that
  is not held this way. At 1 or 0 ticks they may hit that enemy. Movement
  targets, spacing, Rogue prey and Priest healing stay BETA's.
- **Trap order.** A Hunter starts a new Trap only on an enemy with 0 Trap
  ticks. Among the enemies it can Trap now, it Traps the first class in the
  order Priest, Mage, Rogue, Warrior, Hunter (lowest current health, then
  lowest row) and stands still that tick. If it cannot, but its Trap is ready (cooldown 0, not stunned,
  no spawn shield) and such an enemy is in view, it walks toward the first one
  in that order while it keeps using BETA's Basic, with the Trap hold above.
  Otherwise it plays BETA's Hunter and does not Trap; ALPHA's 2-unit Trap rule
  is gone.

By the user's decision of 24 September 2026, BETA and GAMMA are offered for
both teams. Every row of a team that uses GAMMA, Priests included, is recorded
as `tdm_gamma`.

ALPHA v2 and BETA v4 add south-preferred steering around nearby vertical walls,
with north used when the lower passage is unavailable and a turn across the wall
end instead of continued boundary marching. BETA keeps its shoulder screening:
prefer useful wall-phase progress, then distance/action-order ties; if none
exists and an end fits, allow a detour that passes its body-contact check. See
[A35](../design/specification_amendments.md#a35-reactive-tdm-wall-steering) for
the precise local rules. This improves tested wall pockets but does not
guarantee navigation: a three-tick ALPHA allied-congestion stall remains known,
and fixed-side routing may take a longer path. (Superseded, 22 September 2026: this
assessment of ALPHA version 2 and BETA version 4 is historical only; later
study found longer stalls.) Combat and class goals are
unchanged.

ALPHA v3 and BETA v5 fix three ways the controllers froze ([A42](../design/specification_amendments.md#a42-reactive-tdm-fixes-at-blocked-walls)).
A wall end capped by a pillar or a short wall is extended once past the cap, so
the agent steers around the whole shape. Near a wall corner, a move may drift
back by up to one useful stride and still count as progress. A Hunter holds its
distance only while it can shoot the nearest enemy; otherwise it approaches.
ALPHA's ordinary movement does not steer around other bodies, so it can jam.
BETA's Rogue has only the limited body check described above, and only while
it pursues a Priest, Mage or Hunter. Neither controller guarantees a route,
and traffic jams remain.

In a body-blocking study, the defending Warrior can screen a vulnerable Hunter
and provide a moving body around which the Hunter kites, while both maintain
offensive pressure. The specialist can attack that Warrior while pursuing the
Hunter; body avoidance does not imply that it ignores the blocker in combat. For
a TDM finishing scenario, classify the terminal Team A reward, not a particular
movement sequence or minimum attack count: +1 is success, 0 is a draw, and −1 is
failure. A zero reward before termination/truncation is not a draw. Keep
author-observed wins distinct from exact replay-verified witnesses.

To play against a reactive controller:

1. Load a valid scenario or map preview, or keep the default diagnostic arena.
2. Keep Team A Manual and select SharedObs.
3. Select Reactive TDM ALPHA, Reactive TDM BETA or Reactive TDM GAMMA for
   Team B.
4. Stage Team A's actions and Submit; Reset restores the exact starting state.

After editing and saving a scenario, explicitly load its new saved revision to
test that update. Reset restores the already loaded snapshot; it does not pick
up later authoring changes. Saving revised Description/Notes alone does not
change the physical scenario or the controller's behavior identity.

Reactive choices have no task, horizon, roster, score, asset-ID, or respawn
restriction on valid interactive setups. Changing either controller or the
information regime resets the exact loaded snapshot and seed before comparison.
Policy-controlled teams remain inspectable but their action inputs are
read-only; Submit works with both teams policy-controlled. Expected
configuration rejection shows a notice and preserves the session and recording
progress without requiring reconnect. An unexpected internal failure disables
further changes until the session is recovered or restarted.

The old Scripted TDM scorer, separate Reactive MRP/`scenario_1` executable
interface and standalone Scenario 3 controller are removed. Current recorded
controller identities are `reactive-team-deathmatch-controller@3`,
`scenario-5-pressure-controller@5` and
`reactive-team-deathmatch-gamma-controller@2`. Older identities, such as
`reactive-team-deathmatch-controller@2` and `scenario-5-pressure-controller@4`,
remain historical evidence. Private action-source V5 records BETA on Team B,
V6 records GAMMA on Team B, V7 records BETA or GAMMA on Team A (with Team A's
own controller identity), and other interactive combinations retain V4. All
record installed controllers separately from scenario/map identity. Historical
recordings retain their old identities and remain readable; fixed-frame scripted
diagnostics are unchanged. Availability on a setup does not qualify it
scientifically or promise the outcome of another scenario's verified witness.

Loading a scenario does not select its pressure controller automatically, and
saved assets remain controller-independent. Future official evaluations bind the
versioned controller through a separate evaluation definition and apply the same
controller to every treatment and matched-ablation arm. DevClient play remains
diagnostic, not official evidence. See
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations)
and
[A33](../design/specification_amendments.md#a33-one-controller-for-scenarios-3-and-5).
Scenario-derived controller rules are not silently eligible for official
training, and originating-scenario performance is not uncontaminated evaluation.

Saved maps and scenarios do not encode an information regime. Loading one
preserves its authored bytes and binds the selected regime only for that run.

New official evidence must bind `shared_obs`, actor projection
`base-observation-plus-authorized-sensor-source-bank@1`, and the exact
configured-active, same-team, off-diagonal availability matrix on every replay
frame. A successful DevClient run or recording is not by itself official
qualification; the evaluation owner applies the separate official evidence gate.

## Authority and Views

Python owns the environment configuration, JAX state and key, observations,
action masks, staged drafts, hit testing, target mapping, legality, accepted
actions, the single authoritative `step`, canonical transition capture, and the
audience-specific presentation root. The browser owns input capture,
pointer-to-world projection, responsive layout, SVG/HTML paint, panels, help,
and presentation timing. Browser-only activity never advances the simulator.

Oracle View can inspect every authorized actor and stages one independent draft
per active slot. Submit fixes all staged actions before Python applies one
simultaneous turn. Agent POV uses the same global researcher controls and
panels, including the complete roster, target selector, Pending Authorized
Draft, Latest Transition, and joint submission. The selected actor is also the
POV recipient. Only the authoritative battlefield snapshot, hit-testing, ranges,
routes, and choreography are filtered by that actor's fog of war.

The View control switches the displayed audience. Analysis is the only public
presentation: there is no user-selectable density mode. The rendering-only
SharedObs visual-union boundary is defined by [specification amendment
A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It does not provide a materialized SharedObs learner tensor or authorize the
browser to reconstruct geometry, visibility, masks, history, rewards, policy
state, or hidden Oracle facts.

Agent POV also accepts one separate Python-authorized corpse overlay so a dead
body visible to an authorized living sensor remains visible exactly as it is in
Oracle View. That overlay is same-tick, paint-and-inspection-only evidence; it
may additionally admit only a death/respawn presentation cue and that cue's
owned endpoint. It cannot admit another event, move an ability route, or change
policy input, masks, targeting, legality, accepted actions, or simulator
transitions. Every allowed corpse is drawn below every living body, so overlapping
corpses cannot hide living actors or take their mouse targets. Live play keeps
its normal presentation speed.

## Manual Input and Joint Turns

Click the battlefield before using live command keys. Inspector controls and
form fields retain ordinary browser keyboard and Tab behavior.

| Input | Oracle View | Agent POV |
| --- | --- | --- |
| Left click an authorized actor | Control that actor. | Control that visible actor and switch to its POV. |
| Activate a Roster row | Control that actor. | Control any active actor and switch to its POV, whether or not its body is currently visible. |
| Shift+left click an active authorized actor | Select it as the controlled actor's target. | Select that visible actor as the controlled actor's target. |
| Target selector | Stage any globally authorized target. | Stage any globally authorized target. |
| `Escape` | Clear the target and leave battlefield command focus. | Clear the target and leave battlefield command focus. |
| `Tab` / `Shift+Tab` | Cycle active actors without discarding drafts. | Cycle active actors and POV recipients without discarding drafts. |
| `W A S D` / arrow keys | Stage cardinal movement. | Stage cardinal movement. |
| `Q E Z C` | Stage diagonal movement. | Stage diagonal movement. |
| `X` | Stage Stay. | Stage Stay. |
| `0` / `1` / `2` | Stage no combat / Basic / Ultimate. | Stage no combat / Basic / Ultimate. |
| `Space` / `Enter` | Submit every staged actor as one joint turn. | Submit every staged actor as one joint turn. |
| `R` | Reset the arena deterministically. | Reset the arena deterministically. |
| `G` | Toggle Oracle controlled-actor ranges. | Toggle fog-authorized controlled-actor ranges without a simulator command. |
| `?` | Open browser help. | Open browser help. |

Each active actor has an independent movement, target, and combat-lane draft.
Changing the controlled actor does not erase the other drafts. The Pending
Action and legality surfaces show the exact same-decision axis and joint
target/lane result. Latest Transition reports Submitted and Accepted actions
after Python applies the transition.

A second submit is blocked while a request is in flight. If readable
choreography is still active, Submit settles that presentation before sending
the current draft with its expected server revision once. While a live response
is still installing, both views retain at most the first fresh battlefield
staging key and one following fresh Enter, then apply them in that order after
the confirmed update. Once Enter is queued, later staging keys cannot move ahead
of it. Retained input is discarded if the update is stale, fails, requires
reconnection, ends the episode, or shuts down the debugger; a retained Escape
still performs its local focus release without replaying a simulator command.
Animation, hover, panels, help, selection, and filters do not change scientific
authority.

## Inspection and Visual Filters

The selected-target inspector reports authorized identity, relation, distance,
and public geometry. Roster, Comprehensive Agent Class Details, Pending Joint
Action, Latest Transition, Technical Frame, and the target selector remain
available to the researcher in both views. Pending Joint Action reports
movement, ability, target, and exact lane legality without carrying battlefield
anchors. Basic Legality is false when no target is selected. Only SVG-local
inspection and choreography consume fog-filtered geometry. The live Technical
Frame is allowlisted by authority: Episode, Frame, Simulator step, and
conditional Incoming transition. The initial frame has no incoming-transition
row.

Visual Filters contains 20 independently controlled paint families plus Ranges.
Initially enable Ultimate Ability Effects, Spawn Shield, Basic Ability Effects,
Regeneration Effects, Cooldown Effects, Death Effects, Resurrection Effects,
Scrolling Battle Text, Respawn Wave, Death Announcer and Red Zone Floors. Ranges
start off, giving `11 enabled`. Visual Filters and Roster start open; Enable
All selects all 21 controls, and Default Configuration restores these eleven
effects with Ranges off. The complete inventory is:

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
20. Red Zone Floors

Red Zone Floors tints each team's Red Zone, a full-height strip on its own spawn
side, a subtle deep red under obstacles and bodies. The strips come from the
recorded depth and spawn sides; the browser does not work out the rule itself.
Turning the filter off only hides the tint.

Death Announcer names the killing team and shows a compact list of victims.
Hover or focus a victim to see every associated **Kill Contributor**, including
useful same-tick Priest support. Team A stays blue and Team B stays red. The
Death Announcer, respawn banners and scoreboard follow their recorded spawn
sides. Missing or non-left-right spawn banks keep Team A on the left and Team B
on the right. Contributor details use the same accessible
event-information tooltip machinery as Respawn Wave. Missing historical credit
is identified as unavailable rather than guessed.

Notices last 1.5 seconds at normal presentation speed. Enabled cues remain
visible when inspecting their paused/static transition; seek, restart and
replacement clear the previous transition's notices. This global researcher HUD
appears in both POVs and adds no actor observations or battlefield visibility.

Duration Status Badges includes the white crossed-swords **In Combat**
countdown. An overflow badge represents at least two hidden statuses. If only
one remains, its actual status and duration occupy the cell. Compact and
ordinary overflow use the same centered solid badge. Basic Ability Effects and
Ultimate Ability Effects each own their corresponding activation presentation
and damage/healing impact glyphs. Scrolling Battle Text owns the complete
net-health unit: outcome glyph, signed value, recipient label, and connector.
Those parts are enabled or disabled together; damage and healing remain
distinguished by their outcome sign and color rather than by separate filters.

These switches affect browser paint, accessible descriptions belonging to that
paint, and nothing else. They do not redact source data, change simulator state,
or alter authorized event data used by battlefield choreography. Ranges uses its
existing service/local authority path rather than the 20-entry paint schema.
**Enable All** and **Disable All** govern all 20 paint families and the active
Ranges control together.

## Recording and Recovery

Create the destination parent, then opt into recording:

```bash
mkdir -p recordings
./scripts/dev/run_dev_client.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

The target must end in `.marlbg-replay.json`; its parent must already exist, and
the destination must not already exist. The launcher preflights the destination
before building the scenario, discovering runtime provenance, binding a server,
or opening a browser.

Recording retains one authoritative trajectory in memory. Each accepted submit
still performs exactly one transition and one standard capture; there is no
per-transition replay-file write.

- **Finish & Review** closes an open prefix, publishes its self-contained V3
  replay without a metrics sidecar, and changes the same loopback
  page to settled read-only review at frame zero.
- Task termination or the declared horizon closes and saves automatically.
  **Review Replay** performs the frame-zero handoff when requested.
- **Retry save** republishes the exact cached bytes or verifies an already
  successful exact publication; it never overwrites different bytes.
- **Save As** accepts only a new safe replay basename in the original parent.
  It cannot select another directory, traverse paths, follow symlinks, or
  overwrite an existing target.
- Resetting a nonempty recording prefix requires explicit discard
  confirmation. Cancelling preserves the current recording.
- **Exit Combat Debugger** and `Ctrl-C` attempt durable closeout before server
  shutdown. If ordinary publication fails, terminal closeout attempts the
  fixed recovery filename next to the target, reported in the terminal; it never
  overwrites different content.

The page stays online after a persistence failure so bounded recovery actions
remain available. Closing a browser tab alone neither saves nor stops Python.

## Static Reset Snapshot

Render the manual arena's authorized reset state without a browser server:

```bash
./scripts/dev/run_dev_client.sh --static
./scripts/dev/run_dev_client.sh \
  --static --seed 7 --controlled-slot 5 --no-ranges
```

The shell activates the optional `viz` dependency for `--static`. Static mode
creates one reset session, calls the scene-native Matplotlib adapter, registers
no callbacks, and calls `step` zero times. For an exact recorded frame, use the
[Replay Viewer static path](replay_viewer.md#static-matplotlib-frame).

## Loopback Safety

Before opening its port, Python checks the explicit list of allowed browser
assets. The server binds only to `127.0.0.1`, places a random capability token
in the URL fragment, requires the corresponding request header, validates
Host/Origin/ fetch-site metadata, applies a restrictive Content Security Policy
and `no-store`, and handles commands one at a time under a service lock. It
rejects malformed commands, reused command IDs with different contents, and
stale revisions.

Refresh reconnects to the existing session. A stale tab receives the latest
matched frame and presentation. Its submit is not sent again. Use **Reconnect**
after a transient connection loss, and use **Exit Combat Debugger** or `Ctrl-C`
to stop Python.

## Troubleshooting

- **Browser did not open:** use the printed URL or pass `--no-open`.
- **Port unavailable:** omit `--port` or choose another loopback port.
- **Moved-option error:** use the [Replay Viewer](replay_viewer.md) for replay,
  sample, or scripted-scenario work.
- **Inactive controlled slot:** choose a configured-active `arena_5v5` slot.
- **Initial submit is slow:** the initial JAX transition may compile; no warm-up
  transition is hidden from the episode.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

Contributor prose follows the [Documentation
Standard](documentation_standard.md). Return to the [browser-tools migration
page](visual_debugger.md) or the [project README](../../README.md).

Recorded games preserve the captured states, actions, rewards and task outcomes.
Open Replay Viewer detailed analysis to compute the complete metric suite from
those facts and export CSV. New recordings need no metric sidecar; historical V1
replays and their existing sidecars remain readable.
