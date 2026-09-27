# Competitive MARL Workflows

## Write Your Own CTDE Learner

[`examples/own_learner.py`](../../examples/own_learner.py) is a complete small
learner that you own. It builds an actor, a critic and Adam updates, trains
against an ordinary opponent pool, saves two actors, validates them, selects
one by mean point margin and evaluates it on a separate map.

CTDE means **centralized training and decentralized execution**. The actor sees
only each agent's permitted observation and legal actions. The critic may use
privileged physical state during training. The saved actor contains no critic
weights and needs no privileged state to act.

Run a small CPU check from an installed checkout:

```bash
JAX_PLATFORMS=cpu python examples/own_learner.py \
  --output-dir artifacts/own_ctde_check \
  --num-envs 2 --updates 2 --rollout-length 2 --max-steps 3 \
  --custom-reward --fade-rounds 4
```

This collects eight environment transitions and makes two actual updates.
It checks the workflow, not learned skill. `--max-steps` sets the training
game's native horizon. Validation and final evaluation use their normal
300-step games. Their separate batch uses at most 32 environments; this does
not change the training budget. Every output folder must be new; matching names
never authorize replacing a previous experiment.

The same script accepts a larger declared budget without code edits. For
example, 1,024 updates × 32 rounds × 128 environments means 4,194,304
transitions. This is a budget example, not a measured training recommendation:

```bash
JAX_PLATFORMS=cuda python examples/own_learner.py \
  --output-dir artifacts/own_ctde_run \
  --num-envs 128 --updates 1024 --rollout-length 32
```

### Follow One Learning Step

The example uses a small feed-forward actor with shared weights. Applying the
same network to five rows does not let one agent read another agent's private
row. The actor samples movement and a coupled target/Ultimate choice from the
current legal masks. Its `SystemOutput.learning_outputs` carries the probability
of the action from that same call.

The raw loop calls `apply_systems`, `env.step` and `env.reset_done`. Before any
reset, it reads the producing successor through `env.training_state` and
`encode_training_state`. Only the critic receives that vector. The actor update
rebuilds permitted actor inputs from compact saved observations; it does not
receive the critic vector.

One critic predicts the mean reward over active Team A slots. A one-step
actor-critic update uses the native reward plus discounted successor value.
A native win, loss or horizon draw ends the game and contributes no successor
value. Ending a rollout block does not end a continuing game, so that cutoff
keeps its successor value. Dead or inactive actors contribute no actor loss;
a death alone does not end the whole game. This example has empty recurrent
memory because its networks are feed-forward.

The script passes changing weights as numerical inputs to a compiled
collection/update function. It retains compact actor observations and one
critic vector per game. Each update returns small loss, step-count and
same-call probability checks to the host. It copies actor weights only at the
two declared save points. This execution structure is not a GPU speed claim.

### Add A Fading Custom Reward

Without `--custom-reward`, the environment leaves `info.training_facts` as
`None`. Enabling the example's reward requests selected existing Core facts and
adds a small training adjustment for death contributions and new deaths. It
never changes native scores or evaluation rewards.

The public callback shape is:

```python
def reward(before, facts, after, progress):
    # Return a JAX float32 array with ten global agent slots.
    ...
```

`before` and `after` are the public training-only state for the same action.
`facts` belongs to that transition. `progress` is completed training **rounds**,
an int32 scalar shared across a batch. A fade lasting `E` environment
transitions uses `E / num_envs` rounds. The complete function and its vectorized
call are in the example; the signature above is only an interface sketch.

Death facts identify every source that contributed positive effective damage;
they do not choose an exclusive killer. Damage and healing totals are gross
values before health clipping, not realized health loss or gain. Do not infer
other meanings from those fields. Arbitrary shaping can change the learning
objective and has no automatic learning benefit.

### Record A Custom Reward Definition

For an importable callback, `training.shaping.resolve_reward("my_rewards:reward")`
returns the function and a small evidence record. Reuse that function throughout
collection. Resolution imports trusted researcher code, but does not call the
reward. `validate_reward` separately checks its JAX shape and dtype before the
first game step.

The record names the function, available code/default evidence, its source-file
hash and the completed-rounds clock. Save it with your experiment settings.
Changing a constant in that source file changes the evidence too. A hash does
not prove the contents of imported configuration, mutable globals, closures or
external services. Keep those values fixed and record their meaning. Missing
code or source evidence stays explicitly unknown.

A fading reward is one declared definition that reads progress. Its changing
value does not mean the reward definition changed. Rewards are learner feedback;
they do not change the game result used by validation or evaluation.

### Load, Select And Inspect An Actor

Each `actor_step_...` folder contains numerical `actor_weights.npz` and a small
`actor.json` with its measured experience and exact saved-byte identity. These
are this example's actor snapshots, not MARL-BGs learner checkpoints. They do
not resume optimizer or game state. `load_actor(path)` checks the saved bytes
before returning a normal `System`.

The script validates both real snapshots against the same frozen Random panel
on map 42, with both spawn ends. It uses one shared declared confirmation seed
panel and the public `select_checkpoint` point-margin rule. The checkpoint ID and experience added
to selection records come from its own saved snapshots. The final selected
actor plays a separate evaluation on map 47 with a different declared seed.
This is one small validation choice, not broad benchmark qualification.

Read `run_details.json` for the seed, budget, settings and example source hash.
Read `learning.json` for losses, actual collected steps and the probability
check. Read `validation/selection.json` for the selected actor and final result
folder. Load that folder with `marl_bgs.load_results` to inspect episode outcomes,
native points and the shared result views. The ordinary evaluators accept the
saved actor through its own zero-argument factory too:

```python
import json
import os
from pathlib import Path
import marl_battlegrounds as marl_bgs

selection = json.loads(
    Path("artifacts/own_ctde_check/validation/selection.json").read_text()
)
os.environ["MARL_OWN_ACTOR"] = str(Path(selection["selected_actor"]).resolve())
result = marl_bgs.evaluate(
    "examples.own_learner:load_selected",
    "random",
    maps=[47],
    num_episodes=2,
    num_envs=2,
    seed=919,
)
print(result.head_to_head())
```

The selected snapshot may be the earlier one. The code reads the saved decision
instead of assuming that the last update wins. CPU examples use two games; use
an allowed GPU batch, such as 32, for GPU execution.

## Validate The Deployed Team

Use `validate_checkpoint` when the learner controls only some slots. Pass the
same training partners and any separately declared test partners. After running
[the partner example below](#train-selected-slots-with-frozen-partners), this
call reads its actual saved actor path. The saved learner keeps its own checkpoint identity; each evaluated team has its own
ordinary System identity in the saved match records.

```python
import json
from pathlib import Path
from marl_battlegrounds.training import validation

trained = json.loads(
    Path("artifacts/partner_demo/partner_results.json").read_text()
)
panel = validation.create_panel(
    opponents=["random", "tdm-beta"],
    output_dir="artifacts/team_panel",
)
result = validation.validate_checkpoint(
    trained["actor"],
    panel,
    output_dir="artifacts/team_validation",
    partners={
        "ALPHA": "tdm-alpha",
        "Custom Idle": "examples.partner_training:load_idle",
        "New Partner": "random",
    },
    learner_slots=[0, 2],
    system_roster=trained["rosters"]["system"],
    opponent_roster=trained["rosters"]["opponent"],
    partner_labels={
        "ALPHA": "familiar", "Custom Idle": "familiar", "New Partner": "held_out",
    },
    num_envs=32,
)
for partner in result["partner_results"]:
    print(partner["name"], partner["label"], partner["score"])
```

For another run, use the actor path that its training call returned. Partners
may also be live `System` or `Policy` objects or ordinary factory references.
The learner controls the listed physical slots; the partner controls the other
active slots. All methods keep their permitted inputs and their own memory.
Use `system_roster` and `opponent_roster` for a fixed custom team composition.
Their defaults are the canonical five classes.

Every partner plays the same opponent panel, maps and paired game keys. The
combined uncertainty keeps these partner comparisons and both spawn ends in
the same block. `partner_results` also reports each partner separately. A label
records your declared exposure; it does not prove that the partner was held out.
These intervals concern these fixed methods, not variation across training runs.

Built-in training validates every fixed training partner and any declared
`validation_partners`. It fixes validation rosters to the final curriculum
stage, so checkpoints are compared under the same conditions. Random checks
use the same deployed teams. Native game scores remain unchanged by training
rewards. Repeating the same validation call resumes its saved task; changing a
partner, slot assignment, roster or label requires a new output folder.

Saved-record checks bind the learner, declared partner, slots, rosters and
actual match System IDs. They verify those recorded facts; they do not rebuild
unavailable partner weights. Runtime reuse also checks the actual composed
System against the saved match identity.

## Competitive Training Loops

[`examples/competitive_training.py`](../../examples/competitive_training.py)
shows self-play, a named opponent population, a small league rule, population-based
training (PBT), and a response-population loop often called PSRO. These are
ordinary Python loops around the public training and evaluation calls. You can
replace the built-in opponents with your own `System`, `Policy`, saved actor or
trusted `module:function` reference.

Run the complete small CPU example in a new folder:

```bash
JAX_PLATFORMS=cpu python examples/competitive_training.py \
  --output-dir artifacts/competitive_demo
```

It trains two FF-IPPO candidates for eight environment transitions each. It
validates both on the same declared Random confirmation panel, selects by mean
native point margin, continues the winner, trains one new response, and continues
that response after adding its frozen actor to the population. Every training
call saves a real actor and a complete learner checkpoint. The script loads the
selected actor and evaluates it separately on map 47.

These budgets test the workflow. They do not test learned competence. The
population and final demonstration games use the explicit two-step horizon;
confirmation uses normal 300-step games on map 42 with one common paired seed.
Set `--evaluation-max-steps 300` for normal-length population games. Choose and
record an adequate training budget, seeds and evaluation conditions for your own
research. `--num-envs`, `--updates` and `--rollout-length` change that budget through
normal settings. Confirmation uses a separate batch capped at 32, leaving the
training budget unchanged. Use an allowed batch such as 32 for GPU execution; the CPU
example is not GPU cost evidence.

### Choose Opponents In Familiar Python

`TrainConfig.opponents` describes future games. The names `self` and `past` mean
the current learner and eligible frozen past copies. Other names bind to the
methods passed to `train(opponents=...)`. Relative weights are normalized; a
sequence can instead give an exact repeating order. Running games keep the
opponent they started with.

The example's `history_recipe` spells out the requested mixture without making
it a library restriction:

```python
from examples.competitive_training import history_recipe
from marl_battlegrounds import training

settings, methods = history_recipe([
    "/saved/actor_one", "/saved/actor_two", "/saved/actor_three",
    "/saved/actor_four", "/saved/actor_five", "/saved/actor_six",
])
config = training.TrainConfig(
    method="ff_ippo", num_envs=32, total_env_steps=120_000_000, **settings
)
result = training.train(config, output_dir="artifacts/my_population", opponents=methods)
```

This gives 60% current self-play, 20% past copies, and 20% split equally among
ALPHA, BETA and the six named actors. It requests a capture every 5,000,000
transitions and keeps twenty eligible past copies. Before a copy exists, past
share uses the current learner. The small executable demo uses `keep_past=0`:
its tiny budget cannot meet a meaningful safe capture gap. A requested interval
uses the first completed update at or after that many transitions since the
previous actual capture. The saved copy records its actual step. The literal
5,000,000 request works with every supported batch; no manual rounding is needed.

The `pfsp(stats, env_steps)` callback is a small example of prioritized fictitious
self-play: give more future games to named opponents the learner beats less
often. It reads actual cumulative wins, draws and losses from the shared
statistics. A draw counts as half a win. An unseen opponent starts with an
explicit 50% prior, not a claimed measurement. The callback keeps 20% current
self-play and shares 80% among named opponents. This is a separate example rule;
it does not preserve the fixed 60/20/20 recipe above.

The library owns game counts, saved opponent identities, current game assignments
and callback records. You own the rule. A callback changes future choices after
an update. Appending a new member happens at a declared `extend_training`
boundary, where existing member identities are checked again.

### Continue A Winner With PBT

The example declares its confirmation panel before training either candidate.
Both use the same maps, opponents, spawn pairing, seed panel and selection rule.
`select_checkpoint` chooses the measured winner. PBT then calls
`extend_training(winner_full_checkpoint, changes=...)` with new future learning
rates and entropy weight.

That full checkpoint preserves optimizer state, recurrent memory, live games,
random streams and absolute progress. An actor export alone cannot perform this
continuation. The parent stays unchanged and the child gets a new run folder.
The example does not claim the perturbed child is better; that needs another
declared comparison.

### Add A Trained Response And Measure Again

The response loop plays every ordered pair in the current population, including
self-pairs. Each cell has both spawn ends. It reads native Team A and Team B
points from saved episode rows. A missing cell is an error; a self-pair is never
silently filled with zero. This matters because a finite self-pair can have a
nonzero measured margin.

A short host calculation called fictitious play uses that complete table to
produce an empirical opponent mixture. It starts with one count per member and
runs 200 declared steps; ties choose the first listed member. This finite
calculation is not an equilibrium proof. The example trains a fresh finite-budget
response against the mixture, saves its actor, adds that frozen actor, and plays
the expanded table again. A trained neural response is not assumed to be an
exact best response.

Finally, the example extends the response's full checkpoint with the expanded
population and its new mixture. Existing live games continue unchanged; only
later game starts use the new choice rule. The empirical games guide this loop,
so they are adaptive development evidence, not a held-out benchmark score.

`decisions.json` links the selected checkpoint, PBT child, response actor, next
child and final evaluation. Each `population_*/payoffs.json` records population
order, exact System IDs, measured cells, mixture inputs and evaluator paths.
The normal evaluator records retain the actual rosters, map, spawn ends and
games. Different display names do not replace those identities. Keep live
external methods fixed during a table; the example rejects a method whose
recorded identity changes between cells. Identity evidence cannot certify
unreported changes inside an opaque external service.


## Train Selected Slots With Frozen Partners

[`examples/partner_training.py`](../../examples/partner_training.py) trains
physical slots 0 and 2 while a frozen partner controls the remaining active
slots. Team A is Mage, Warrior, Mage, Priest, Hunter; Team B is Warrior, Mage,
Warrior, Hunter, Priest. These repeated classes are an explicit one-stage
curriculum on training map 0. Its default partner pool contains ALPHA and a
small custom idle `System` from the example. Both use the same ordinary partner
interface.

```bash
JAX_PLATFORMS=cpu python examples/partner_training.py \
  --output-dir artifacts/partner_demo \
  --method ff_ippo --parameter-sharing none
```

`--method` accepts `mappo`, `ippo`, `ff_mappo`, `ff_ippo`, `qmix` and `pqn_vdn`.
`--parameter-sharing all` uses one actor network, `class` uses one per class,
and `none` uses one per physical slot. The two Mages share weights in `class`
mode and have separate weights in `none` mode. Each actor keeps its own
recurrent memory when the method is recurrent. A class change at a future game
reset chooses the matching class network; physical slot mode keeps that slot's
network. Both nonshared modes store five actor networks and optimizer states,
so they use more model memory. This does not establish better learning.

Partners receive no optimizer update. Their physical observations and actions still affect the game; only the learner's
owned active slots contribute to its learning targets and statistics.

The default makes one tiny learning block with two environments. PPO and QMIX
collect four transitions; PQN first collects its three required initial rounds,
then one two-round learning block, for ten transitions. The example uses a
constant PQN rate so this short run does not end its rate schedule immediately.
These are software checks, not training recipes or evidence of learned skill.

Before training, the script declares a fixed Random validation panel with
separate routine and confirmation seeds. The runner validates both deployed
teams on maps 42–46 with normal 300-step games and one seed pair per map. It
selects by native point margin. One learning block leaves one eligible trained
candidate; longer runs use the same selection owner.

The script saves a full checkpoint and actor exports. It loads the selected
actor from `result.selected_actor`, then calls
`team(actor, partner, slots=[(0, 2), (1, 3, 4)])` separately for every training
partner. The export contains the learner alone; reassembling the team preserves
the intended ownership. Final evaluation keeps the declared repeated-class
rosters and uses held-out map 47, a separate seed, both spawn ends and an explicit
two-step demonstration horizon. Use `--evaluation-max-steps 300` for normal-length
final games. That setting does not shorten validation.

Python callers can pass their own named partner mapping and physical
`learner_slots` to `run`. A repeating order chooses partners only when games
start. The learner's self-play mirror also keeps frozen partners in its other
slots. `partner_results.json` links the saved learner and each composed team's
ordinary evaluation records, including the selection record, exact identities,
rosters and spawn coverage. Declared training membership is not proof that every member was sampled in a
short run. Inspect the saved assignments for actual exposure; this example makes
no zero-shot coordination claim.
