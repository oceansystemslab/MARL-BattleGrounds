# MARL-BattleGrounds

MARL-BattleGrounds is a JAX-native environment suite and benchmark for
heterogeneous adversarial multi-agent reinforcement learning.

The project is currently under development.

## DevClient

Use the DevClient as the developer workspace for live combat debugging and
minimal reusable map and Team Deathmatch scenario authoring:

```bash
./scripts/dev/run_dev_client.sh
```

Its three areas are Combat Debugger, Maps, and Scenarios. The Combat Debugger
provides staged simultaneous actions, exact legality, Oracle and authorized
agent-POV views, saved-scenario loading, browser-local visual filters, and
optional replay recording. Maps and Scenarios save revisioned local drafts with
visible paths, validate them through the existing simulator authorities, and
delete unwanted saved assets after confirmation. Saved scenarios load
directly; saved maps can be opened as clearly labelled deterministic
default-5v5-TDM previews without modifying the map. Either team can remain
manual, use the scripted Team Deathmatch controller, or use the built-in Random
controller for same-start testing. Random samples only the current valid action
support and is available under both SharedObs and NoSharedObs.
Every applicable selector shows all latest saved revisions in numeric-aware
asset-ID order, with native scrolling for longer lists. New Map and Scenario
asset IDs use lowercase snake case, such as `tdm_map_10`, while their visible
names remain free-form. Save is explicit and persists numbered local revisions
under ignored `artifacts/dev_client/` storage across DevClient restarts; the
DevClient does not autosave.
Existing replay artifacts remain outside the DevClient.

Record one manual episode to a canonical replay and adjacent metric sidecar:

```bash
mkdir -p recordings
./scripts/dev/run_dev_client.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

See the [DevClient and Combat Debugger guide](docs/dev/combat_debugger.md) for
launch options, authoring, saved-scenario loading, input, authority,
recording/recovery, static reset snapshots, visual filters, and troubleshooting.

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
eight whole-clock rates, 18 paint-filter families plus Ranges,
provenance-bearing PNG export, and researcher-space metric download across
visual POVs. It cannot stage or submit simulator actions.

See the [Replay Viewer guide](docs/dev/replay_viewer.md) for artifact selection,
sample provenance, scenario isolation, transport and keyboard behavior,
audience boundaries, export, static rendering, and troubleshooting. If you
used the historical combined command surface, start with the concise
[browser-tools migration page](docs/dev/visual_debugger.md).

The DevClient and Replay Viewer use tracked native HTML, CSS, SVG, and
JavaScript served by base Python dependencies. Researchers need neither Node.js
nor npm. The browser installs only Python-authorized presentation data; the rendering-only
SharedObs visual-union boundary is recorded in
[specification amendment A17](docs/design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).

## Actor information regimes

The policy/evaluation layer supports homogeneous `shared_obs` and
`no_shared_obs` execution through one reference-rollout lifecycle. NoSharedObs
remains a first-class mode selected with the same
`execution_information_mode` setting and reported separately. The current
low-level rollout requires that setting explicitly; the DevClient introduced
under A20 activates SharedObs as the first researcher-facing default.
SharedObs composes already-authored teammate sensor rows without changing the
simulator, and evaluation records store base observations plus exact source
availability rather than a second materialized tensor. See
[specification amendment A19](docs/design/specification_amendments.md#a19-sharedobs-structured-runtime-advancement)
for the runtime and Milestone 12 ownership boundary. Every researcher-facing
launch surface must default to SharedObs, while requiring an explicit mode at
the low-level API prevents an old caller from silently changing scientific
regime.

## Static snapshots

The optional `viz` dependency is activated automatically by either shell
launcher when `--static` is present:

```bash
# One manual arena reset snapshot.
./scripts/dev/run_dev_client.sh --static

# One exact frame from an immutable replay.
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 0
```

Static rendering starts no browser server. Install the optional dependency
explicitly for direct Python use with `uv sync --extra viz`.

## Design and evaluation

- [Specification amendments](docs/design/specification_amendments.md) are the
  controlling public authority for accepted departures from the historical
  design PDF.
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
