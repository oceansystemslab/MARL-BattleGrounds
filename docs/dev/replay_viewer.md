# Replay Viewer

The Replay Viewer is the read-only browser product for immutable semantic
replays, checked demonstration samples, and scripted demonstrations
materialized as replay bundles. It always uses the fixed Analysis presentation
and cannot stage actions, submit a simulator transition, reset an episode, or
record to a user-selected destination.

Manual live work belongs to the [Combat Debugger](combat_debugger.md).

## Select exactly one input

Every invocation must choose exactly one artifact, sample, scripted scenario,
or list operation:

| Selector | Purpose |
| --- | --- |
| `--replay PATH` | Validate and open a local canonical replay bundle. |
| `--sample-replay NAME` | Verify and open one checked-in sample by stable launch name. |
| `--scenario NAME` | Materialize one registered scripted demonstration in isolation, then open its validated bundle. |
| `--list-scenarios` | List default scripted demonstrations and exit. |
| `--list-sample-replays` | List checked-in sample names and descriptions and exit. |

Open a local artifact:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json
```

Choose the initial frame and audience without opening the browser
automatically:

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

List and materialize a scripted demonstration:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The public browser options are `--frame-index`, `--pov-slot`,
`--view oracle|pov`, `--ranges`/`--no-ranges`, `--port`, and `--no-open`.
`--seed` applies only to scenario materialization. List operations reject
unrelated options. `--static` has its own narrow matrix described below.
Option abbreviations are rejected.

## Scripted-scenario isolation

`--list-scenarios` shows researcher scripted demonstrations and omits the
manual `arena_5v5` laboratory. `stacked_team_auras` follows `aura_crossfire` in
the default catalog and demonstrates simultaneous Basics under two same-team
Mage and two same-team Warrior emitters per team.

Developer visual-stress demonstrations are excluded by default:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios --include-stress
./scripts/dev/run_replay_viewer.sh \
  --scenario max_status_stack --include-stress
```

`--include-stress` is only a catalog-discovery/authorization input for
`--list-scenarios` or `--scenario`; it is not a browser selector and is
unavailable with local artifacts or checked samples.

Scenario materialization runs in a temporary child process with
`JAX_PLATFORMS=cpu`. The child executes the registered commands, records a
metric-complete replay/sidecar pair, validates both through the public loader,
and exits. Only that immutable loaded bundle crosses into the Replay Viewer.
The read-only viewer process does not import or run simulator control.

## Checked sample replays

The checked V1 bundle under `examples/replays/v1/` contains exactly three
replay/metric pairs plus `manifest.json`:

| Launch name | Source scenario | Coverage focus |
| --- | --- | --- |
| `death-respawn-shield` | `death_respawn_cycle` | Lethal damage through the first post-shield interaction. |
| `recovery-status-lifecycle` | `recovery_refresh_cycle` | Recovery, rejection, refresh, break, reapply, and expiry. |
| `mirrored-five-class-ultimates` | `mirrored_ultimates` | Reciprocal demonstrations of every class Ultimate family. |

Every sample uses an 18×12 map. The manifest records stable names, source
scenarios, transition/frame counts, event-kind coverage, byte lengths, hashes,
and actual source/runtime provenance. Samples are deterministic unofficial
presentation demonstrations—not benchmarks, policy evaluations, source-tree
attestations, or host attestations.

Verify the complete seven-file set through the public loaders:

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory examples/replays/v1
```

Generation is maintainer-only and refuses overwrite. Generate into one new,
absent directory only after scenario source, tests, and public documentation
are frozen; verify the complete set before an explicit reviewed publication.
Never hand-edit one member or delete the checked directory merely to bypass the
no-overwrite boundary.

## Transport and exact-frame summaries

| Control | Behavior |
| --- | --- |
| **Start** / **End** | Seek to the first or final captured frame. |
| **−10** / **−1** / **+1** / **+10** | Issue one clamped absolute seek. |
| **Play** / **Pause** | Serialize playback with at most one replay request and one presentation in flight. |
| Frame slider | Preview the target tick locally without a request; commit one exact absolute seek when the value is committed. |
| Tick label | Show the authoritative current and final simulator ticks joined from the recorded timeline. |

Button seeks, committed slider seeks, view/range changes, reconnects, and other
non-play navigation install a settled exact-frame summary. The summary retains
the enabled durable and incoming-transition cues together without animation or
hidden timer state. Pressing Play restarts the displayed incoming transition at
logical time zero when one exists, waits for its scaled presentation to settle,
and then requests the next frame. Each accepted exact successor advance may
animate only that successor's recorded incoming transition.

The eight supported rates are exactly **0.25×, 0.50×, 0.75×, 1.00×, 1.25×,
1.50×, 1.75×, and 2.00×**. A rate scales the complete presentation clock,
including animation phases, waits, and the replay terminal hold. It never
changes simulator ticks or artifact contents.

### Document keyboard shortcuts and exclusions

Unmodified Left, Right, and Space are document-level shortcuts for previous,
next, and play/pause. They do not require the battlefield or timeline to be
active. The viewer deliberately does nothing when:

- Shift, Control, Alt, or Meta is held;
- Space is an auto-repeat event;
- the event originates in or under a button, input, select, textarea, link,
  disclosure summary, dialog, editable region, or ARIA widget such as a
  slider, textbox, combobox, spinbutton, or menu item; or
- replay authority is absent, offline, hidden, or not yet installed.

These exclusions preserve ordinary browser and assistive-technology behavior.

## Audiences and recorded authority

Oracle View exposes the full authorized replay presentation, Reference
selection, authorized ranges, completion/processing truth, PNG provenance, and
metric availability. Agent POV keeps one fixed recipient and receives only its
authorized NoSharedObs or recorded SharedObs visual-union projection.
Reference/inspection changes are local presentation choices and never mutate
the artifact.

The SharedObs visual union follows
[specification amendment A11](../design/specification_amendments.md#a11-sharedobs-recorded-visual-union-presentation).
It may join only recorded same-decision-epoch rows from authorized same-team
sensor sources. It does not recompute geometry, visibility, line of sight,
masks, mechanics, or state; include teammate masks/history, rewards, policy or
critic state, transition facts, or hidden Oracle truth; or claim to be a
materialized SharedObs learner input.

### Technical Frame

Replay Oracle uses an exact five-leaf Technical Frame allowlist:

1. Artifact digest prefix
2. Frame
3. Simulator step
4. Incoming transition, omitted at frame zero
5. Ordinary movement distance scale

Agent POV receives only Frame, Simulator step, and its conditional authorized
Incoming transition. It never receives the canonical artifact digest or
movement scale through this panel. Completion and Processing remain distinct
rollout/host-processing facts on their authorized replay surface; neither is a
Technical Frame expansion.

## Visual filters

Visual Filters contains exactly 23 browser-local paint families, all enabled
by default:

1. Aura Fields
2. Aura Modifier Badges
3. Duration Status Badges
4. Spawn Shield
5. Rejected Action Feedback
6. Basic Ability Effects
7. Ultimate Ability Effects
8. Damage Effects
9. Healing Effects
10. Regeneration Effects
11. Cooldown Effects
12. Charge Movement
13. Status Application
14. Status Reapplication
15. Status Refresh/Extension
16. Natural Status Expiry
17. Freezing Trap Break
18. Status Clear on Death
19. Death Effects
20. Respawn Wave
21. Resurrection Effects
22. Spawn-Shield Expiry
23. Scrolling Battle Text

Duration Status Badges includes the white crossed-swords **In combat** countdown.

A filter change pauses playback and reinstalls the current settled summary
after filtering, so disabled paint never consumes layout space. Filters do not
change authorized data, the Latest Events feed, or the separate Ranges state.
**Restore All** enables all 23.

## PNG export and metrics

**Export PNG** is enabled only when one coherent replay frame is connected,
visible, settled, and free of pending replay/presentation work. It exports the
battlefield alone—not the toolbar, timeline, or inspectors—at exactly twice
its displayed pixel dimensions. The result uses the bundled fonts and locked
battlefield background, reflects the current audience, selection, ranges, and
23 filter states, and embeds one canonical
`MARL-BattleGrounds Replay Provenance` iTXt record. Export does not navigate the
replay or request another replay frame.

**Download Metrics** is Oracle-only and enabled only at the same settled
boundary. Activating it requests the canonical adjacent metric report once and
downloads the exact bytes when available. A missing sidecar reports absence
without a download. Agent POV is rejected from audience alone: it cannot
request the metric endpoint or learn availability, a path, or a filename.

## Static Matplotlib frame

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

## Loopback and artifact safety

Local artifacts, checked samples, and materialized bundles pass whole-artifact
and companion validation before the server binds or a browser is opened.
Invalid schemas, canonical bytes, hashes, event/frame joins, frame indices,
POV recipients, symlinks, and unsupported paths fail closed.

The browser server binds only to `127.0.0.1`, uses a random fragment-delivered
capability token, validates request headers and origins, serves an explicit
asset allowlist under restrictive Content Security Policy and `no-store`, and
serializes revisioned/idempotent commands. Replay commands change only the
selected recorded authority; none can call `step` or modify the source files.

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
- **Metrics unavailable:** confirm Oracle View and a verified adjacent metric
  sidecar; Agent POV cannot query availability.
- **PNG export disabled:** pause playback and wait for the exact-frame summary
  to settle in a visible connected tab.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

The replay format itself is documented in
[replay_format.md](../evaluation/replay_format.md). Return to the
[browser-tools migration page](visual_debugger.md) or the
[project README](../../README.md).
