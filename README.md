# MARL-BattleGrounds

MARL-BattleGrounds is a JAX-native environment suite and benchmark for
heterogeneous adversarial multi-agent reinforcement learning.

The project is currently under development.

## Development Visual Debugger and Analyzer

Open the deterministic Milestone 1–5 Visual Debugger and Analyzer in a local
browser:

```bash
./scripts/dev/run_debug_renderer.sh
```

Python owns the simulator, session, legality, commands, and transitions. It
returns an allowlisted JSON frame; the browser constructs the SVG/HTML
presentation and presentation-only motion. The launcher always prints a
loopback URL and also attempts to open it automatically. Running the analyzer
requires neither Node.js nor npm.

For a one-frame Matplotlib snapshot instead:

```bash
uv sync --extra viz
./scripts/dev/run_debug_renderer.sh --static
```

A validated semantic replay can be rendered at an exact recorded frame without
starting the simulator or a browser:

```bash
./scripts/dev/run_debug_renderer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 0
```

See the [Visual Debugger and Analyzer guide](docs/dev/visual_debugger.md) for
controls, joint-turn planning, view authorization, scenarios, replay/static
rendering, visual semantics, troubleshooting, and contributor checks.

## Design and evaluation

- [Specification amendments](docs/design/specification_amendments.md) record
  proposed departures from the original design document and state their
  activation status.
- [Evaluation metric specification](docs/evaluation/metric_specification.md)
  defines metric meanings, attribution limits, scorecard surfaces, and
  candidate dispositions.
- [Evaluation protocol](docs/evaluation/protocol.md) defines evaluation cells,
  aggregation, inference, cross-play, scenarios, failure handling, and
  information-regime reporting.
- [Standard replay format](docs/evaluation/replay_format.md) defines the
  versioned semantic replay normal form, content-addressed companion artifacts,
  whole-artifact validation boundary, and canonical bounded local persistence.

## Author

MARL-BattleGrounds is developed by Ulixes Tariq Hawili as part of the SPADS CDT,
University of Edinburgh School of Engineering, and the Ocean Systems Lab at HWU.

## License

Licensed under the Apache License, Version 2.0.
