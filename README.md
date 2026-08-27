# MARL-BattleGrounds

MARL-BattleGrounds is a JAX-native environment suite and benchmark for
heterogeneous adversarial multi-agent reinforcement learning.

The project is currently under development.

## Combat Debugger

Use the Combat Debugger for manual work in the live 20×10 `arena_5v5`
laboratory:

```bash
./scripts/dev/run_debug_renderer.sh
```

It provides staged simultaneous actions, exact legality, Oracle and authorized
agent-POV views, browser-local visual filters, and optional replay recording.
The product always uses the fixed Analysis presentation and does not host
scripted demonstrations or existing replay artifacts.

Record one manual episode to a canonical replay and adjacent metric sidecar:

```bash
mkdir -p recordings
./scripts/dev/run_debug_renderer.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

See the [Combat Debugger guide](docs/dev/combat_debugger.md) for launch
options, input, authority, recording/recovery, static reset snapshots, visual
filters, and troubleshooting.

## Replay Viewer

Use the separate read-only Replay Viewer for immutable artifacts, checked
samples, and materialized scripted demonstrations:

```bash
./scripts/dev/run_replay_viewer.sh --replay episode.marlbg-replay.json
./scripts/dev/run_replay_viewer.sh --list-sample-replays
./scripts/dev/run_replay_viewer.sh --sample-replay death-respawn-shield
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The viewer validates or materializes one complete replay bundle before opening
the browser. It offers settled exact-frame summaries, serialized playback,
eight whole-clock rates, 24 local visual filters, provenance-bearing PNG
export, and Oracle-only metric download. It cannot stage or submit simulator
actions.

See the [Replay Viewer guide](docs/dev/replay_viewer.md) for artifact selection,
sample provenance, scenario isolation, transport and keyboard behavior,
audience boundaries, export, static rendering, and troubleshooting. If you
used the historical combined command surface, start with the concise
[browser-tools migration page](docs/dev/visual_debugger.md).

Both browser products use tracked native HTML, CSS, SVG, and JavaScript served
by base Python dependencies. Researchers need neither Node.js nor npm. The
browser installs only Python-authorized presentation data; the rendering-only
SharedObs visual-union boundary is recorded in
[specification amendment A11](docs/design/specification_amendments.md#a11-sharedobs-recorded-visual-union-presentation).

## Static snapshots

The optional `viz` dependency is activated automatically by either shell
launcher when `--static` is present:

```bash
# One manual arena reset snapshot.
./scripts/dev/run_debug_renderer.sh --static

# One exact frame from an immutable replay.
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 0
```

Static rendering starts no browser server. Install the optional dependency
explicitly for direct Python use with `uv sync --extra viz`.

## Design and evaluation

- [Specification amendments](docs/design/specification_amendments.md) record
  accepted departures and clarifications relative to the original design.
- [Evaluation metric specification](docs/evaluation/metric_specification.md)
  defines metric meanings, attribution limits, scorecard surfaces, and
  candidate dispositions.
- [Evaluation protocol](docs/evaluation/protocol.md) defines evaluation cells,
  aggregation, inference, cross-play, scenarios, failure handling, and
  information-regime reporting.
- [Standard replay format](docs/evaluation/replay_format.md) defines the
  versioned semantic replay normal form, whole-artifact validation boundary,
  companion artifacts, and canonical bounded local persistence.
- [Dependency policy](docs/dev/dependency_policy.md) separates researcher
  runtime requirements from optional and contributor-only tooling.

## Author

MARL-BattleGrounds is developed by Ulixes Tariq Hawili as part of the SPADS CDT,
University of Edinburgh School of Engineering, and the Ocean Systems Lab at HWU.

## License

Licensed under the Apache License, Version 2.0.
