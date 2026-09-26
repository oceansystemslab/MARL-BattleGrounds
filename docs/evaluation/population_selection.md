# Select A Population

Declare the candidates and selection rule **before** their validation tournament.
The tool then selects the requested number of entrants from that complete field.
It writes the chosen members before refitting their ratings or running test games.
It does not admit an official release, train a model or replace failed games.

The initial official field excludes scripted controllers and LLMs. Those methods
remain supported in local experiments. The tiny example below uses built-ins to
check the software; it does not choose the official field or prove strong play.
See [the tournament protocol](protocol.md#big-12-tournament-and-baseline-library)
for the scientific requirements and historical version-1 rules.

## Declare And Run The Field

```python
import marl_battlegrounds as marl_bgs

field = {
    "entrants": ["random", "tdm-alpha", "tdm-beta"],
    "maps": [42, 43, 44, 45, 46],
    "games_per_opponent": 10,
    "max_steps": 2,
    "selection": {
        "size": 2,
        "entrant_order": ["random", "tdm-alpha", "tdm-beta"],
        "failure_policy": "require-complete-field",
    },
}
result = marl_bgs.run_tournament(config=field, output_dir="selection-games")
decision = marl_bgs.select_initial_population(result, output_dir="population")
```

Use new or empty output folders. The tournament prepares and freezes controller
identities, map contents, rules, random streams, schedule and selection rule in
its version-2 configuration. It saves that configuration before any game. The
selection function also accepts a saved run path or `load_results()` result.
It reads the same records without loading the models or playing more games.

The `selection` field accepts a mapping or a JSON file. Relative file paths use
the containing declaration's directory. Its fields are:

| Field | Meaning |
| --- | --- |
| `stage` | `"validation"` by default. Only registered validation maps may be used. |
| `size` | Required integer from two through the complete candidate count. |
| `entrant_order` | Each method name once, in cutoff tie order. Omission uses the input entrant order. |
| `failure_policy` | `"require-complete-field"` by default. Every scheduled game is required. |

The field must have equal opponent weights. The existing joint Elo fit ranks
all candidates together. A rating tie uses expected score against that same
field; an exact remaining tie uses `entrant_order`. The tool uses saved numbers
without rounding or a hidden tolerance. Weak valid entrants stay eligible.
No method family receives a reserved place.

Resume inherits the saved declaration. A supplied different declaration fails.
A separate `challenger=` or a changed game budget cannot change a selection field
after its declaration. Earlier games cannot be relabelled as a new selection
stage. Ordinary tournaments without `selection` retain their existing behavior.

## Read The Decision

A complete decision writes `population.json` with `status="complete"`, exact
`members`, full-field `rankings`, the frozen rule, source identities and a content
hash named `population_id`. The released size comes from its frozen member list.
This hash detects changed content; it is not an official approval signature.

An interrupted or failed field writes `status="incomplete"`, candidate identities,
source status and missing game IDs. It selects no smaller substitute field and
adds no replacement seeds. Resume the original tournament, then create a new
decision in a new output folder. Keep the incomplete record for inspection.

The optional `declaration=` argument to `select_initial_population` only checks
that a supplied rule matches the one already saved before games. It cannot add
selection rules to an ordinary completed tournament.

The tool reuses the checked full-field ratings. After writing membership, it
uses the existing rating fitter on the selected entrants' validation games and
writes a separate `refit.json`. Selecting the whole field reuses its existing
fit. Refit order can change, but membership cannot. A solver failure is recorded
in `refit.status`; it leaves the frozen members intact. The returned dictionary
includes that separate refit report. File or record errors are raised, and any
already written population decision remains on disk.

Saved sampling facts travel with the decision and refit. Unknown determinism
stays unknown. Repeated deterministic games do not become independent evidence,
and fixed-policy intervals do not measure differences between training seeds.

## Run The Separate Test Stage

A selection test stage requires the complete frozen decision. It checks member
identities before opening the run and permits only registered test maps. Its
records are separate from the validation games. Test ratings cannot select the
population again.

```python
if decision["status"] == "complete":
    test_field = {
        "entrants": [
            row["controller"]["reference"] for row in decision["members"]
        ],
        "maps": [47, 48, 49, 50, 51],
        "games_per_opponent": 10,
        "max_steps": 2,
        "selection": {
            "stage": "test",
            "population": "population/population.json",
        },
    }
    test_result = marl_bgs.run_tournament(
        config=test_field, output_dir="population-test"
    )
```

This example's members came from string references. A saved
selection does not serialize an arbitrary live session. This declared workflow
uses reusable entrant references; ordinary list tournaments still accept live
Systems. Ordinary custom evaluation
on test maps remains supported outside this declared selection workflow.

## Use The CLI

Save the first example's `field` mapping as `field.json`, then run:

```bash
python -m marl_battlegrounds tournament --config field.json --output-dir selection-games
```

The tournament prints `Run Directory: selection-games/<run-id>`. Use that exact
child folder in the next command; replace `<run-id>` with the printed ID:

```bash
python -m marl_battlegrounds select-population "selection-games/<run-id>" --output-dir population
```

The selector needs the folder containing `run_details.json`, not its parent.
The second command prints the decision as JSON. It exits with code 1 for an
incomplete field. `--declaration rule.json` checks an already bound rule. The CLI
calls the same public selection function as Python.

For a complete small CPU demonstration, including the test stage:

```bash
python examples/population_selection.py --help
python examples/population_selection.py --output-dir artifacts/population-demo
```

The example defaults to CPU, two concurrent games and two ticks per game. These
settings exercise the software cheaply; they are not scientific budgets or GPU
throughput measurements. A scientific campaign must declare its final draw rule,
search effort, independent seeds, selection conditions and budgets before use.
