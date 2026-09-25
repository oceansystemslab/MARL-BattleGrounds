# Browser Tools Migration

Use **DevClient** for live play and local map/scenario editing. Use **Replay
Viewer** for recorded games, checked samples, and scripted demonstrations. These
replace the former combined Visual Debugger and Analyzer.

| Product | Main Work | Launcher |
| --- | --- | --- |
| [DevClient](combat_debugger.md) | Combat Debugger, Maps, and Scenarios | `./scripts/dev/run_dev_client.sh` |
| [Replay Viewer](replay_viewer.md) | Inspect, play, analyze, and export recorded games | `./scripts/dev/run_replay_viewer.sh` |

Both use the fixed Analysis presentation. Python owns simulation, action
legality, saved-content validation, and the facts each view may show. The
browser owns layout, help, filters, and animation. Browser-only interaction does
not run a simulator turn.

## Command Migration

Run these commands from the checkout with `uv` available. Replace `PATH`,
`NAME`, and `N` with a real replay path, listed name, and frame index.

| Previous Task | Current Command |
| --- | --- |
| Open the developer workspace | `./scripts/dev/run_dev_client.sh` |
| Record a live episode | `./scripts/dev/run_dev_client.sh --record-replay PATH` |
| Draw a live reset snapshot | `./scripts/dev/run_dev_client.sh --static` |
| Open a replay formerly selected with debugger `--replay` | `./scripts/dev/run_replay_viewer.sh --replay PATH` |
| Open a checked sample formerly selected with debugger `--sample-replay` | `./scripts/dev/run_replay_viewer.sh --sample-replay NAME` |
| Open a scripted demonstration formerly selected with debugger `--scenario` | `./scripts/dev/run_replay_viewer.sh --scenario NAME` |
| List scripted demonstrations | `./scripts/dev/run_replay_viewer.sh --list-scenarios` |
| List checked samples | `./scripts/dev/run_replay_viewer.sh --list-sample-replays` |
| Draw one exact replay frame | `./scripts/dev/run_replay_viewer.sh --replay PATH --static --frame-index N` |

`run_debug_renderer.sh` is a compatibility redirect to DevClient. Replay Viewer
has no authoring area, action composer, manual submit, reset, or recording
destination. Its Left/Right/Space shortcuts navigate recorded frames and yield
to ordinary form controls. Node.js and npm are contributor tools; they are not
needed to launch the built browser assets.

## Live Editing and Controllers

Create or open a map/scenario in DevClient, save it, then load that revision in
Combat Debugger. A saved scenario restores its authored state. A saved map opens
a clearly labeled default 5v5 TDM preview. A map that retains an approved TDM
map's source name and content digest is recorded under that map's registered
identity. **Open in Debug** uses the same Python build-and-validation path. A
rejected load leaves the current session unchanged. Reset restores the already
loaded snapshot; it does not pick up later edits to the saved asset.

Save is explicit, with no autosave. It creates numbered local revisions and
checks the expected revision before writing. Selectors list every applicable
latest revision in numeric-aware asset-ID order. Asset IDs use lowercase snake
case; new obstacle IDs use `obstacle_N`. Visible names remain free-form.
**Delete Saved** needs confirmation and removes all revisions of that identity;
an open draft remains available as an unsaved copy.

Both teams can use **Manual**, **Reactive TDM ALPHA**, **Reactive TDM BETA**,
**Reactive TDM GAMMA** or **Random**. Reactive controllers require SharedObs;
Random and Manual also support NoSharedObs. Policy-controlled agents stay inspectable,
but their actions cannot be edited manually. Submit can run both automatic
teams. A controller or information-mode change resets the loaded snapshot.

ALPHA v3 handles all five classes. BETA v5 uses the same rules except that Rogue
pursues observed living Priests, then Mages, then Hunters, with lowest-health
and global-slot tie breaks. Its glancing shoulder routes and local wall steering
do not guarantee navigation. Both prefer passing below nearby vertical walls,
then above when needed, and steer past a wall end capped by a pillar or short
wall. A Hunter holds its distance only while it can shoot the nearest enemy
([A42](../design/specification_amendments.md#a42-reactive-tdm-fixes-at-blocked-walls)). A three-tick ALPHA allied-congestion stall remains
known. (Superseded, 22 September 2026: this assessment of ALPHA version 2 and BETA
version 4 is historical only; later study found longer stalls.) The shared `scenario_5` BETA behavior serves Scenarios 3 and 5; the old
standalone Scenario 3, Scripted TDM, and separate Reactive MRP interfaces were
removed. Historical recording identities remain readable.

GAMMA v2 is BETA plus three rules for Warriors, Mages, Hunters and Rogues: with
no enemy in view they walk toward the middle of the enemy spawn pads; they never
damage an enemy with 2 or more Hunter Trap ticks left; and Hunters start a new
Trap only on an untrapped enemy they can reach now, choosing Priest, Mage,
Rogue, Warrior, Hunter in that order. Its Priests keep BETA's rules
([A43](../design/specification_amendments.md#a43-reactive-tdm-gamma)).

These are diagnostic/scenario-pressure tools, not official baselines or Big 12
entrants. Future official evaluation definitions must bind the same versioned
controller to every compared treatment; saved physical scenarios stay
independent of controllers. See the [live workflow and detailed controller
rules](combat_debugger.md#loading-saved-scenarios-and-map-previews),
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations),
and
[A32](../design/specification_amendments.md#a32-scenario-5-shoulder-bypass-and-fallback-prey).

## Display and Evidence Boundaries

The 20 visual filters and separate Ranges control change display, not simulation
or recorded data. SharedObs display follows
[A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation):
it combines permitted same-tick sensor views for rendering. It does not create a
learner input or grant access to hidden facts. A successful DevClient session or
a viewable replay does not by itself qualify an official experiment.

Follow the [Documentation Standard](documentation_standard.md) when changing
these tools. Use [Quality Gates](quality_gates.md) for contributor checks and
[Dependency Policy](dependency_policy.md) for the runtime/tooling split.
