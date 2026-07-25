# Milestone 5 visual debugger

The Milestone 5 visual debugger is an explicit-submit development tool for
inspecting the simulator's battlefield geometry, action mask, accepted action,
combat state, and selected-target facts. It uses ordinary public `reset` and
`step` calls. It does not add debugger behavior to the simulator core.

The renderer is stateless. Keyboard and mouse input, deterministic scenarios,
transition comparison, logs, and transient-effect history live under
`scripts/dev/visual_debugger`.

## Setup and launch

Install the development and optional visualization dependencies:

```bash
uv sync --extra viz --extra dev
```

Run the default 5v5 laboratory:

```bash
./scripts/dev/run_debug_renderer.sh
```

The shell launcher resolves the repository root from its own path, so the
command also works when invoked from another current working directory.

Useful invocations:

```bash
./scripts/dev/run_debug_renderer.sh --help
./scripts/dev/run_debug_renderer.sh --list-scenarios
./scripts/dev/run_debug_renderer.sh --scenario basic_support
./scripts/dev/run_debug_renderer.sh --scenario status_stack --verbose
./scripts/dev/run_debug_renderer.sh --scenario ultimate_showcase --static
./scripts/dev/run_debug_renderer.sh --controlled-slot 5 --no-ranges
```

`--static` renders the deterministic reset snapshot without registering input
callbacks and without calling `step`.

## Command-line options

| Option | Meaning |
|---|---|
| `--scenario NAME` | Open a registered scenario. Default: `arena_5v5`. |
| `--list-scenarios` | Print names, modes, and descriptions, then exit without importing Matplotlib. |
| `--seed N` | Set the deterministic reset/step key seed. Default: `0`. |
| `--controlled-slot N` | Initially control an active fixed global slot instead of the scenario default. |
| `--static` | Draw only the reset snapshot; do not register callbacks. |
| `--verbose` | Print the expanded geometry, mask, aura, speed, and episode diagnostics. |
| `--ranges` / `--no-ranges` | Show or hide the controlled actor's range circles. Ranges are shown by default. |

An unknown scenario, invalid argument, or inactive controlled slot exits with
code `2`. A missing `uv` executable exits with code `127`. A missing Matplotlib
dependency exits with code `2` and prints the exact installation command.

## Controls

| Input | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle forward/backward through active global slots. |
| Left click | Select an active global target for inspection. |
| `Shift` + left click | Immediately control the clicked active actor while preserving the selected global target. |
| Right click / `Escape` | Clear the target to target-none. |
| `1` | Explicitly arm Basic lane 0, even when the pair is unavailable. |
| `2` | Explicitly arm Ultimate lane 1, even when the pair is unavailable. |
| `W A S D` | Select cardinal movement. |
| `Q E Z C` | Select diagonal movement. |
| `Space` / `Enter` | Submit the manual action, or the next frame in a scripted scenario. |
| `N` | Submit the next registered reference/script frame. |
| `R` | Reset the current scenario deterministically. |
| `Shift+R` | Leave the session unchanged and explain why cooldown-only clearing is unavailable. |
| `G` | Toggle the controlled actor's range circles. |
| `V` | Toggle concise/verbose transition logs. |
| `[` / `]` | Switch to the previous/next scenario. |
| Close window | Disconnect callbacks, restore Matplotlib key mappings, and exit. |

There is no simulation timer. Selecting or directly controlling an actor,
changing movement, arming a lane, cycling the controlled actor, toggling
presentation, or redrawing never advances the simulator or consumes a PRNG
key. After native Tab traversal, the application schedules canvas-focus
restoration so the first following keyboard command is not lost to a toolbar
button.

`Shift+R` does not clear cooldowns. Reset and step are the only public
state-to-observation/mask snapshot boundaries, and no public coherent
snapshot-rebuild API exists for a cooldown-only edit. Use `R` for a full
deterministic reset.

## Pending-action behavior

The reset pending action is Stay, target-none, and automatically armed Basic
lane 0. A clicked target is retained even when no combat lane is available.
Basic auto-arms only when the exact current lane-0 value for that actor-target
pair is true. Ultimate never auto-arms.

`Shift` + left click changes control without stepping or splitting the key. It
preserves the selected global target, resets movement to Stay, discards
explicit arming, converts the target to the new actor's relative category, and
auto-arms Basic only when that exact lane-0 pair is currently available. If the
clicked actor is also the selected target, the relation becomes `self` and the
reticle remains visible.

Keys `1` and `2` deliberately permit an unavailable in-domain pair to be
submitted. This lets the authoritative simulator demonstrate canonical
rejection. The debugger does not pre-filter it.

Mage Burst is a target-none Ultimate. Pressing `2` while controlling a Mage
clears a selected target and arms lane 1. It creates no target link or
Ultimate-radius circle. It can still be armed during cooldown or stun so its
rejection can be inspected.

After every submitted transition:

- the selected global target persists;
- movement resets to Stay;
- Ultimate disarms;
- Basic re-arms only if successor lane 0 is available for the retained target;
- otherwise the target remains selected with no lane armed;
- distance, relation, line of sight, visibility, ranges, and lanes are derived
  again from the successor decision epoch.

At termination or truncation, submissions are blocked before key splitting.
Inspection, actor cycling, presentation toggles, reset, and scenario switching
remain available.

## HUD and selected-target inspector

The side panel reserves 42% of the 17×9 figure and has six stable sections:

1. Play-by-play
2. Controlled agent
3. Selected target
4. Pending action
5. Latest accepted result
6. Technical details and visual key

Human-facing sections use identities such as `TEAM A MAGE (id_0)` and the
ability names `BASIC`, `BURST`, `CHARGE`, `TRAP`, `POISON`, and `HOLY WORD`.
Raw `gN` and `tN` categories appear only in monospaced technical diagnostics.

The selected-target section reports relationship, center distance, line of
sight from the public geometry authority, observer-relative visibility from
the current observation, observation-radius membership, and exact Basic and
Ultimate availability. Target-none displays an explicit no-target state.

The pending-action section reports the chosen movement, ability, global target,
lane-0/lane-1 availability, and exact pair legality. Lane values come directly
from
`select_target_use_ultimate_joint_mask[actor, target, lane]`; the debugger
never reconstructs them from geometry. Displayed distance, LOS, visibility,
and range facts are not claimed as the cause of an unavailable pair.

While a target is selected, non-visible active agents are dimmed above their
class fill but below protected health, aura, team, lane, reticle, identity,
status, and transient artists. Clearing the target clears this visibility
treatment. Visibility is supplied to the renderer as presentation data; the
renderer does not derive it.

## Battlefield legend

### Durable snapshot presentation

| Meaning | Presentation |
|---|---|
| Mage / Warrior / Hunter / Rogue / Priest | Class-filled body and `M/W/H/R/P` label using aqua, brown, green, mustard, and pink. |
| Team A / Team B | Blue/red circumference at the true physical body radius. |
| Health | Inset green/amber/red annulus; it never changes the collision outline. |
| Controlled actor | Observation/Basic/Ultimate ranges and the controlled-agent HUD section; no extra body chevrons. |
| Selected target | Magenta crosshair and bullseye. |
| Lane 0 / lane 1 | Labelled lower-left green `0` and lower-right violet `1` arcs; unavailable lanes are gray and the armed lane is thicker. |
| Observation / Basic / Ultimate range | Gray dashed, green solid, and violet dotted world-space circles. |
| Persistent status | Detached neutral rounded chips above the body, in stable two-column semantic order, with source-class border accents. |
| Mage aura | Inset upper cyan dotted band with a strong edge and under-stroke. |
| Warrior aura | Inset lower bronze hatched/dashed band. |
| Identity | Class letter plus `id_N`; raw `gN`/`tN` values are technical diagnostics only. |

Detached status chips use point offsets, so they do not change world geometry,
collision, or hit testing. Their labels and remaining submitted-step durations
are `CHARGE-STUN`, `TRAP`, `POISON-STUN`, `CHARGE-SLOW`, `HUNTER-SLOW`,
`POISON-SLOW`, `ANTI-HEAL`, `FREEDOM`, and `BURST`. They deliberately avoid
circular outlines that could be mistaken for collision radii.

### Transient presentation

| Event | Presentation and lifetime |
|---|---|
| Net health change | One signed red/green public net delta for one submitted transition. |
| Basic damage | Source-colored action link and red impact flash. |
| Basic healing | Priest-colored link and one green recipient pulse. |
| Holy Word | Strong link and two green/pink recipient pulses. |
| Mage Burst activation | Short labelled `BURST!` flash. |
| Warrior Charge | Short labelled `CHARGE!` flash plus the public before/after trail. |
| Hunter Trap | Short labelled `TRAP!` flash. |
| Rogue Poison | Short labelled `POISON!` flash. |
| Holy Word | Short labelled `HOLY WORD!` flash. |
| Rejection | Red `M×`, `C×`, or `TUPLE×`; rejected combat also has a dashed target link, and simultaneous components share one label. |

One-step visuals expire on the next submitted transition. Charge traces are
shown at opacity `1.00`, `0.65`, and `0.35`, then removed. Redrawing does not
age effects. Reset and scenario switching clear all transient history.

Health labels show only `after_health - before_health`. Accepted effect labels
identify capability activations, but do not invent gross per-source damage or
healing when effects aggregate, clip, or cancel.

## Deterministic scenarios

Every scenario builds a validated `EnvConfig`, uses ordinary `reset`, sets
`ordinary_movement_distance_scale=1.0`, and starts at maximum health with zero
cooldowns, statuses, and action history.

### `arena_5v5`

Interactive `18×12` laboratory with all five classes per team, a pillar at
`(9,3)` with radius `0.9`, and a rotated `3×0.5` wall at `(9,7.8)` with angle
`0.45`. Use it to inspect open/blocked LOS, observer visibility, relations,
range boundaries, collision, and pair masks.

### `basic_support`

Two scripted frames. The first applies three simultaneous Basics and exposes
Mage amplification, Warrior mitigation, Hunter slows, and exact health
successors. The second combines Priest self-healing with simultaneous Hunter
damage, retaining the public net `0.00` without fabricating gross
contributions.

### `ultimate_showcase`

Three scripted frames. A Mage Basic prepares the allied Hunter; then Mage
Burst, Warrior Charge, Hunter Trap, Rogue Poison, and Holy Word activate
simultaneously. The final Hunter Basic demonstrates an unambiguous Trap break,
status decrements/expirations, and cooldown decrement.

### `aura_crossfire`

One scripted reciprocal Hunter-Basic frame while both Hunters are inside Mage
amplification and Warrior mitigation. Both health values end at `92.18`, both
aura bands remain visible, and both Hunter slow channels apply.

### `status_stack`

Four scripted frames that combine Charge plus precommitted movement, Trap,
Poison, Priest healing/Freedom, a stunned recipient holding Stay, Trap break,
Freedom speed floor, stacked slow movement, duration expiration, and cooldown
decrement. A Charge-plus-movement trail is labeled as combined realized
displacement; it does not claim a private intermediate landing.

The intentionally rejected movement/combat boundary remains covered by
test-only fixtures. Every command in these five user-facing scenarios is
preflighted at its actual decision epoch and must be in-domain, exact-mask
legal, and accepted as submitted.

## Terminal diagnostics

Concise and verbose modes both print readable play-by-play first and a stable
technical block second:

```text
PLAY-BY-PLAY
  TEAM A PRIEST (id_2) had a net health change of 0.00 HP.
  Accepted contributors included TEAM A PRIEST (id_2) BASIC and
  TEAM B HUNTER (id_7) BASIC.
  The public transition does not expose the gross damage/healing split.

TECHNICAL DIAGNOSTICS
  Transition   scenario=basic_support step=1 -> 2 terminated=0 truncated=0
  Actor id_7 [g7] submitted move=Stay[0] target=t8 ultimate=0
               accepted  move=Stay[0] target=t8 ultimate=0
               mask move=1 lane0=1 lane1=0 pair=1 domain=1
  Health id_2 92.00 -> 92.00 net=+0.00
```

Verbose mode adds target relation/distance, geometry, visibility, position,
aura, speed, and reward records per report actor. Both modes print named
channels and scalar facts, not raw tensor dumps.

Accepted actions come only from the successor state's previous-action fields.
Movement and combat-pair acceptance are reported independently. Rejections show
domain and current-mask facts, never guessed causal explanations.

Health narration reports only the observed net loss, gain, or zero change.
Accepted contributors and same-epoch public multipliers may be listed, but the
debugger never fabricates gross per-source damage/healing when effects are
simultaneous, clipped, cancelled, or mitigated. Newly applied Burst, Freedom,
and other statuses are described as successor state, not as mechanics that
governed the transition that applied them.

Status changes are conservatively classified as application, refresh,
decrement, expiration, unambiguous Trap break, or unclassified clear. A Trap
transition from `1` to `0` alongside accepted damage remains unclassified
because natural expiration and break cannot be distinguished from public
artifacts.

## Reset and reproducibility

The CLI seed initializes a master JAX PRNG key. Reset and scenario switching
reconstruct the same deterministic key order. Each successful submission splits
the current key once and calls `step` once with the action mask paired to the
current state. UI-only actions never split the key.

Scripted frames can command multiple actors. Manual mode builds a joint action
for only the controlled actor. Both paths meet at the same submission boundary;
scripted frames do not pass through the single-actor pending-action builder.

## Architectural boundary and non-goals

Reusable code under `marl_battlegrounds.rendering` draws snapshots and
already-described cues. It has no callbacks, timers, simulator calls, inferred
event model, or history.

The debugger's `SelectedTargetFacts`, before/after transition comparison, and
transient aging are development diagnostics. They are not a production replay
event schema and are not serialized. An eventual replay viewer may naturally
reuse the stateless battlefield renderer and visual descriptions while
consuming recorded state or event data through its own durable contract.

The debugger does not:

- modify simulator action, observation, reward, status, or termination
  semantics;
- duplicate LOS, visibility, or action-mask algorithms;
- attribute an unavailable action to one displayed geometric fact;
- recover gross health contributions from an aggregated net transition;
- reconstruct the private intermediate Charge landing;
- create a replay timeline or production event protocol.
