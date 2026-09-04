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
controller for same-start quality-control testing. Random samples only the
current valid action support, is available under both SharedObs and
NoSharedObs, and is a diagnostic controller rather than an official baseline.
The generic Scripted TDM controller is likewise debugging and regression
tooling: neither controller is a Big 12 baseline or a deterministic scenario
pressure controller.
Every applicable selector shows all latest saved revisions in numeric-aware
asset-ID order, with native scrolling for longer lists. New Map and Scenario
asset IDs use lowercase snake case, such as `tdm_map_10`, while their visible
names remain free-form. Save is explicit and persists numbered local revisions
under ignored `artifacts/dev_client/` storage across DevClient restarts; the
DevClient does not autosave.
Existing replay artifacts remain outside the DevClient.

Record one manual episode to a canonical-format replay and adjacent metric
sidecar:

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

Paper 1 uses one canonical actor-information contract: `shared_obs` with actor
projection `base-observation-plus-authorized-sensor-source-bank@1`. Every frame
of official evidence must record exactly the configured-active, same-team,
off-diagonal source-availability matrix derived from the frozen roster.
SharedObs composes already-authored teammate sensor rows without changing the
simulator; evaluation records store those base observations and the exact
availability authority rather than a second materialized tensor.

The generic policy/evaluation layer and replay readers continue to support
homogeneous `no_shared_obs` execution through the same lifecycle for
diagnostics, custom research, and historical compatibility. NoSharedObs is not
an official baseline or Paper 1 comparison axis. Saved maps and scenarios are
regime-independent; official execution binds canonical SharedObs only after an
asset is loaded and validated. See
[specification amendment A25](docs/design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution)
for the full eligibility and compatibility boundary.

The benchmark standardizes authorized actor inputs, categorical identities,
domains, sentinel meanings, shapes, masks, slot/source identities, provenance,
and reference-encoder conventions. It does not mandate one-hot encodings,
embeddings, attention, or any other researcher neural architecture.

## Planned scenario evaluations and Big 12

The planned public scenario suite is primarily a controlled behavioral-ablation
instrument. Each official evaluation definition will compare a complete method
with its matched ablation under the same scenario revision, embedded map and
initial state, deterministic reactive pressure controller, canonical SharedObs
contract, seeds and sides, training budget, checkpoint-selection rule, and
primary endpoint. Scenario pressure controllers will follow versioned rules and
authoritative action masks rather than replaying hard-coded action tapes. They
remain separate from the saved, controller-independent DevClient scenario and
from the generic Scripted TDM and Random diagnostic controls. Scenario results
provide evidence for a specific behavioral claim under frozen conditions; they
do not contribute to Elo or establish general strength. See
[specification amendment A26](docs/design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations).

Public evaluation scenarios and their complete content closure must not inform
training, checkpoint selection, early stopping, hyperparameters, prompts,
curricula, population weights, or any other adaptive choice. Future M11/M12
pipelines will enforce content-addressed training, validation, and evaluation
manifest separation, while official systems retain complete provenance for
maintainer reproduction. This is a reproducibility and eligibility boundary,
not a claim that open-source software can make deliberate misconduct
impossible.

The planned Big 12 is a rolling ladder of exactly twelve method-level entrants,
each represented by one validation-selected fixed tournament system and one
Elo. The tentative initial roster is:

1. RNN-IPPO, parameter-shared
2. RNN-MAPPO, parameter-shared
3. RNN-MAPPO, class-specific actors
4. RNN-HAPPO
5. HyperMARL-PPO
6. RNN-QMIX
7. RNN-PQN-VDN
8. MAPPO-PFSP League
9. MAPPO-PSRO
10. S\*-Curriculum
11. S\*-Curriculum-Shaped
12. Qwen-Five

This roster and its training/evaluation pipeline are planned, not implemented.
Rows 1–11 will each retain three independently trained runs and their checkpoint
histories, but only the fixed checkpoint selected by the frozen validation-only
rule enters the tournament. Qwen-Five remains tentative until a measured
throughput and resource gate is passed. Twelve systems yield 66 unordered
pairings and, at 100 episodes per pairing, 6,600 tournament episodes. The raw
win/draw/loss matrix remains authoritative; rating implementation details must
pass their own pre-tournament gate.

Weekly Big 12 reviews will publish immutable dated snapshots. A qualified new
method may enter by relegating the lowest method; a week without a qualified
challenger changes nothing. The Paper 1 snapshot remains frozen, and
pool-centred Elo values from different weekly pools are not directly
longitudinally comparable. Current and former entrants remain reproducibly
available in a cumulative Baseline Library through immutable manifests—never a
mutable `latest_big_12` alias. Scenario pressure controllers, generic Scripted
TDM, Random, and internal training-population members are not Big 12 entrants.
See
[specification amendment A27](docs/design/specification_amendments.md#a27-rolling-big-12-and-baseline-library-governance).

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
  canonical actor-information provenance.
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
