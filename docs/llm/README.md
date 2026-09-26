# LLM Systems, Game Views And Actions

An LLM can act through the ordinary `System` interface. Start a compatible
model server separately, then create a System:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm

system = llm.make_system("my-served-model", "http://127.0.0.1:8000/v1")
result = marl_bgs.evaluate(
    system, "random", num_episodes=2, maps=[0], num_envs=2,
    max_steps=300, output_dir="runs/llm-example",
)

print(result.table("episodes"))
print(llm.call_summary(result))
print(result.paths)
```

Replace `my-served-model` with the name your server accepts. The evaluator owns
the managed client and closes its connections and workers, including after an
error. Reusing the System opens a fresh client when needed. It never stops
the independent server. Construction does not download weights, make a model
request or start a server. The simulator needs no new transport dependency.

The default request uses vLLM's Chat Completions and `/tokenize` routes, with
thinking off, temperature 0 and a 64-token reply allowance. Use
`server_type="chat"` for standard Chat Completions fields. That mode needs a
trusted `token_counter` for history; without one it requires `history_turns=0`
and leaves context rejection to the server. A Chat Completions-compatible
endpoint is required; this is not a client for every provider's native API.

Try the complete script:

```bash
JAX_PLATFORMS=cpu python examples/llm.py --model my-served-model \
  --format default --history-turns 2 --output-dir runs/llm-example
```

Use `JAX_PLATFORMS=cuda,cpu` for a GPU simulator with CPU host helpers, after
checking memory available beside the model server. Choose ordinary `num_envs`
for the simulator and `Client(concurrency=...)` for simultaneous HTTP work.
The client default is 16 requests, a 60-second complete request deadline and
no retries. The deadline applies to each HTTP call, not a whole actor decision
or game; fitting history can make several tokenizer calls. Both teams can borrow one client and share that limit. Finished
actor requests free their slots without waiting for an earlier slow reply;
the System returns only after the full team decision is ready.

Use the [qualified Qwen recipe](qwen_recipe.md) for pinned server commands,
measured costs and the fixed 4B/9B comparison.

See [custom formats](custom_formats.md) to change the prompt and use a non-JSON
reply while keeping the same history, scheduling and action checks.

For a caller-owned connection pool, use `with llm.Client(URL) as client:` and
pass `client=client` to `make_system`. Runners leave supplied clients open.
This also lets two Systems share one request limit. For a hand-written loop,
keep that client context open, or enter `system.resource_scope(False)` around
the complete loop. A managed System cannot make unowned requests.

## Use A Factory From The Package Command

A factory is an ordinary function that takes no arguments and returns a System.
The example provides one for each reply format. Examples are not included in
the installed wheel or source distribution. Run from a repository checkout,
or copy the example into your own importable module. From the repository root:

```bash
export MARL_LLM_MODEL=my-served-model
export MARL_LLM_URL=http://127.0.0.1:8000/v1
export MARL_LLM_REVISION=my-pinned-model-revision
export MARL_LLM_HISTORY=2
JAX_PLATFORMS=cpu python -m marl_battlegrounds evaluate \
  --system examples.llm:make_default_system --opponent random \
  --episodes 2 --maps 0 --max-steps 300 --num-envs 2 \
  --output-dir runs/llm-factory --save-replays 2
```

Use `examples.llm:make_custom_system` for the two-word parser from the
[custom-format tutorial](custom_formats.md). Both factories use the shared
loader and evaluator. They make no requests during construction. The runner
owns the managed client. The URL defaults to `http://127.0.0.1:8000/v1`, history
defaults to 0, and an omitted revision means unknown model content. Set the real
revision when comparing or resuming experiments.

To resume, replace `--output-dir` with `--resume-from` and the saved run directory
printed by the command. Keep the model, revision, history and format settings
unchanged. Completed games need no new model requests. Changes to the declared
method are rejected before play.

## Run Tournaments

Pass the same System to the ordinary tournament functions:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm

system = llm.make_system("my-served-model", "http://127.0.0.1:8000/v1")
result = marl_bgs.run_tournament(
    [system, "random", "tdm-alpha"], maps=[0], episodes_per_pair=2,
    num_envs=2, max_steps=300, output_dir="runs/llm-tournament",
)
print(result.table("tournament_rankings"))
print(llm.call_summary(result))
```

A managed client stays open across that System's unfinished matchups and closes
when the tournament exits, including after an error. Supplied clients remain
caller-owned. Each game still starts with fresh actor history. Completed
matchups acquire no client on resume. Use `resume_from=result.run_dir` with the
same methods and settings to resume the saved custom tournament.

For a local canonical challenger, use
`marl_bgs.run_canonical_tournament(system, output_dir="runs/llm-canonical")`.
This requires an installed, verified canonical snapshot and its local record
assets; M11 does not fabricate or publish a release. The
[canonical tournament guide](../evaluation/canonical_tournaments.md) explains
setup. Resume with the same System and `resume_from` pointing to its run.
The challenger remains Team A while the schedule exchanges spawn ends.
Incumbent games retain their original identities and records.

The [custom-format factory](custom_formats.md) works through these same routes.
A tournament keeps a whole System's batch contract; it does not advance separate
actors or lanes while another part of that System is still deciding.

## Start From An Authored State

Use the same System with `evaluate_episodes` and an `EpisodeSpec` whose
`initial_state` is your authored Core state. See the
[evaluation workflow guide](../evaluation/workflows.md) for schedule construction.
The model sees the state's actual game tick. Recorded `decision_step` instead
counts decisions since this scheduled episode began. For example, a start at
tick 5 records its first decision as 0 while the prompt still says tick 5.
History starts empty and then retains only that actor's played turns.

Saved replay transitions contain both teams' submitted world actions. They can
be played again without asking a model. Do not reflect `world_action` a second
time. This reproduces the saved choices; it does not promise that a fresh model
request gives the same answer.

## Read Calls And Recovery

Game tables and model calls have different jobs. `llm.call_summary(result)`
works with a fresh result or `marl_bgs.load_results(run_dir)`. It returns:

- `all_attempts`: recorded calls, tokens and failures from all execution attempts.
- `completed_games`: counts only from the attempt that saved each completed game.
  Report both teams' fallback counts beside those games' win rates.
- `by_system`: the same two scopes for each saved System registration ID, with
  its recorded name. This keeps ownership clear when tournament entrants change
  between Team A and Team B. These IDs are not canonical entrant IDs.

These summaries read the selected run's own passes. Canonical games reused from
older runs keep their original records; their model calls and fallback counts
are excluded here. Include that source evidence before reporting costs or
fallback counts for the whole tournament population.

Token totals cover measured usage only. `missing_usage_replies` marks replies
with incomplete usage. Abrupt interruption can lose unflushed costs; these
summaries are not a spending guard.

`records="light"` is the default. Rows name the actor, team, game,
turn and execution attempt. They keep the request hash, reply and checked actions
when available. Forced choices have no request or reply; a failed choice may
have no action.
Use `records="full"` to retain the exact serialized request too, including
custom prompts and fitted history. A hash cannot reconstruct omitted text.
`records="none"` skips call files, prompt hashing and reply copies; small failure
and cost counters remain available. Unsaved runs also keep only these counters.

```python
for call in llm.read_calls(result.run_dir):
    print(call["episode_id"], call["team"], call["actor"],
          call["outcome"], call.get("world_action"))
```

Call rows say `played`, `abandoned` or `unknown`. Played means the shared game
step completed. It does not mean the game was saved or the ability hit. A failure
before a joint step abandons both teams' answers and proposed history. A process
interruption may leave unknown outcomes or no final row.

The existing game writer alone decides which games are safely on disk. Its
atomic update saves completed-game IDs with their execution-attempt bindings.
Call files are flushed before that update. A crash between saving a game and
writing its separate call-file acknowledgement is resolved from `run_details.json`.
Completed games are skipped on resume. Unfinished games restart with fresh
history and a new attempt ID. Earlier call files remain available. `chunk_size`
limits when finished games reach the writer; its `buffer_size` separately limits
when completed rows reach disk. Neither setting saves every in-flight request.

## History And Failures

History is off by default. `history_turns=2` keeps at most two earlier living
turns for each actor. The entries contain that actor's permitted view, masks and
submitted action. They survive death and respawn and clear when the game resets.
Dead actors make no request. A living actor with exactly one legal move and one
legal combat pair also needs no model request; its played choice still counts
in history. Histories stay separate even when both teams share a client.

The token counter counts the wrapped request and reserves reply space. If needed,
the adapter removes oldest whole history entries from that request. It never
shortens the current view, rules or legal menu. The parser receives exactly the
history used by its request. Retained memory and a request's included history
are separate: a later, smaller view may fit more retained entries.

Errors stop execution by default. `failure_policy="fallback"` permits a checked
Stay/no-combat choice for declared malformed or illegal replies and transport
failures. Configuration errors, context overflow, local cancellation and custom
code bugs still stop. A parser returning the wrong native shape or type is a
code bug. A valid native action forbidden by the current masks is an illegal
reply. Fallback accounting and model-call records must be interpreted with the
runner's outcome records; game CSV files alone do not explain provider failures.

The HTTP deadline covers waiting for a request slot, DNS, connection, TLS,
response reading and decoding. It cannot forcibly stop arbitrary Python custom
functions or prove that the remote server stopped generating. Close interrupts
owned HTTP work and waits up to its cleanup grace, default 2 seconds. It raises
if user work survives, retaining that work in the closed client.

## Standalone View And Action Helpers

Turn one actor's permitted input into text, then check its named action before
submitting it. The helpers run in Python on the host. They do not load a model,
start a server or change the simulator.

```python
import jax
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm
from marl_battlegrounds.policies.input import (
    mirror_move,
    mirror_team_view,
    team_on_right,
)

env = marl_bgs.make(
    "tdm", map_id=0, metrics="none", balance_spawn_locations=False
)
observations, state = env.reset(jax.random.key(1))
inputs = env.policy_inputs(observations, state, team=0)
actor = jax.tree.map(lambda value: value[0, 0], inputs.actors)
world_masks = jax.tree.map(lambda value: value[0, 0], inputs.action_mask)

# Show this actor the game with its spawn bank on the left.
reflected = team_on_right(actor)
view, masks = mirror_team_view(actor, world_masks, reflected)
view, masks = jax.device_get((view, masks))
prompt = llm.format_actor_view(view, masks)
print(prompt)

# This fixed reply makes the example runnable without a model server.
reply = '{"move":"stay","combat":"no_combat"}'
action = llm.parse_action_reply(reply, masks)
action = action._replace(move=mirror_move(action.move, reflected))
action = llm.validate_actor_action(action, world_masks)

# Use legal sampled actions for the other nine actors.
actions = env.sample_actions(jax.random.key(2), state)
actions = actions._replace(
    move=actions.move.at[0].set(action.move),
    select_target=actions.select_target.at[0].set(action.select_target),
    use_ultimate=actions.use_ultimate.at[0].set(action.use_ultimate),
)
observations, state, reward, done, info = env.step(
    jax.random.key(3), state, actions
)
```

Each actor must receive its own permitted input. Returning five actions for a
team does not permit sharing private inputs or memory between its actors.
`policy_inputs` applies the supplied source permissions before delivery. The
formatter expects that filtered input; it is not a replacement permission checker.

## Named Replies

The default reply is one JSON object with exactly two string fields:

```json
{"move": "north", "combat": "enemy_2_ultimate"}
```

`legal_action_names(masks)` returns the legal strings under `move` and `combat`.
Each listed move can accompany each listed complete combat choice. The menu
contains only current legal choices; their names keep the same meaning each turn.

| Name | Native Meaning |
| --- | --- |
| `stay`, `north`, `south`, `east`, `west`, `northeast`, `northwest`, `southeast`, `southwest` | Core movement categories 0–8 |
| `no_combat` | No target, no Ultimate: `(0, 0)` |
| `ultimate` | Untargeted Ultimate: `(0, 1)`, when legal |
| `ally_0_basic` … `ally_4_basic` | Ally target categories 1–5, no Ultimate |
| `enemy_0_basic` … `enemy_4_basic` | Enemy target categories 6–10, no Ultimate |
| `ally_0_ultimate` … `enemy_4_ultimate` | The named target with Ultimate use |

Names come from Core's existing action table. `enemy_2` means the same unit as
“Enemy 2” in the clients: enemy roster row 2, native target category 8. Targets
are stable roster slots, not a list sorted by distance. A target's number does
not change when it becomes hidden or dies.

The parser rejects duplicate or extra keys, unknown names, wrong types, fenced
answers and surrounding prose. Whitespace around the JSON is fine. It never
repairs a reply, retries a request or silently submits another action.

`ReplyFormatError` means the reply does not match the format.
`IllegalActionError` means the decoded native action is invalid or masked.
Malformed input masks raise `ValueError`. Combat legality uses the complete
**target/Ultimate pair**. Two allowed marginal categories can form an illegal pair.

`validate_actor_action(action, masks)` is also available to custom parsers. It
checks integer scalar categories before narrowing them to int32, then checks the
original masks. Booleans, floats, oversized integers and batched actions fail.
All successful replies return the existing `ActorAction` with three host NumPy
int32 scalars.

## What The Text Contains

The default includes every supplied field family:

| Input | Text Representation |
| --- | --- |
| Public rules and obstacles | Rules first, then obstacle rows in their original order |
| Context | Current tick, horizon, dimensions, task flags, scores, thresholds and Red Zone depth |
| Self, allies and enemies | Exact unit facts, stable roster assignments and separate visibility |
| Public spawn lifecycle | Pads, shields, wave rules/clocks, membership, alive flags and classes |
| Previous accepted actions | Movement, target and Ultimate in the observer's roster names |
| Shared sensor sources | Separate source permission, visible candidates, unit facts and objectives |
| Reserved objectives | Positive-zero default; any nonzero supplied value remains explicit |
| Masks | Legal moves and complete combat pairs; marginal combat masks are their unions |

Core's feature constants own the field names. Unit features include current
position, health, speed and status, plus the amounts, ranges and durations the
unit can apply. `capability_*` describes possible effects; other status fields
describe effects currently on the unit. See the [Core input contract](../../src/marl_battlegrounds/core/types.py)
for the simulator interface.

The text omits positive-zero fields and padding under an explicit zero rule.
Identical unit rows share one definition, but each observer/source assignment
and visibility flag remains. Different facts never share a definition. There is
no shared mutable prompt cache.

“Hidden” differs from a visible zero row. “Unavailable” differs from an allowed
source that sees nothing. An all-zero previous-action row means “Not seen.”
An accepted Stay/no-combat action is a real one-hot history entry. Previous enemy
targets already use the observing actor's ally/enemy names; they are not swapped
again, and a previously selected target may now be hidden.

Decimal text recovers the original float32 value, including signed zero. Integer
lifecycle values retain their full int32 value. This exactness applies to the
**supplied view**: reflecting coordinates already carries float32 rounding.
The shared obstacle reflection also uses its documented matching tolerance.
Neither exact text nor a legal menu proves that an action will hit or move.

## Coordinate Frame And Cost

`format_actor_view(..., frame="left")` labels already-transformed input. It does
not reflect it. Use `team_on_right` and `mirror_team_view` first, then undo the
selected movement exactly once with `mirror_move`. The reflection flag comes
from the public spawn bank; crossing the map does not change it. Target IDs
never reflect. `frame="world"` is available when the supplied view is unchanged.

The formatter accepts one actor, not a whole team or batch. It checks shapes,
dtypes and finite values. It calls `jax.device_get` before processing values;
callers can transfer a whole batch once and pass individual NumPy rows. Do not
place this Python formatter inside the compiled simulator loop.

## Your Own Format

The text layout and JSON syntax are supplied conveniences. Researchers may
choose a different presentation, select different permitted fields or use a
different reply syntax. A custom decoder should return `ActorAction`, then call
`validate_actor_action` with retained original masks. Preserve the same actor,
decision and coordinate frame throughout. A format choice does not grant access
to private simulator state or another actor's private input.


## Measured Development Check

On 25 September 2026, Qwen3.5-4B at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` completed two development games on
map 0 against Random. Each game used seed 19671501 and a 300-tick horizon.
The first used the default spawn end and no history; the second used the other
end and two history turns. Both were draws with zero kills and deaths for both
teams. These check the software routes; they do not compare history quality or
establish playing strength.

| Measurement | No History | Two Earlier Turns |
| --- | ---: | ---: |
| Complete Evaluation Time | 262.4 seconds | 529.7 seconds |
| Model Calls | 1,496 | 1,499 |
| Input Tokens | 4,459,359 | 10,536,764 |
| Output Tokens | 18,040 | 18,014 |
| Forced Actions Without A Call | 4 | 1 |
| Reply/Transport Failures | 0 | 0 |
| Played Fallbacks, Team A / Team B | 0 / 0 | 0 / 0 |
| Full Run Records And Replay | 43.4 MB | 63.7 MB |

The model ran on the internal RTX 5090 beside unrelated training; the simulator
ran on CPU. vLLM 0.30.0 used dynamic FP8, thinking off, temperature 0, seed 0,
64 reply tokens, 16 request slots and prefix caching off. The server became
healthy after 122.3 seconds. The whole window, including setup and shutdown,
took 916.8 seconds. One-second samples found peaks of 15,024 MiB owned VRAM and
7.61 GB summed process RSS for the server and worker. RSS counts resident memory
for each process and can count shared pages more than once. Samples can miss brief peaks; timings include
GPU contention and are not isolated speed measurements. These CPU simulation
runs have no GPU simulator-transfer measurement.

Full records retain exact requests and cost much more disk than the default
light records. A separate fixed-fake check of 32 games for 8 ticks wrote zero
call bytes with records off, 0.813 MB with light records and 16.19 MB with full
records. All three produced the same game results. Its single warmed timings
were 6.25, 7.51 and 6.59 seconds; that noise does not establish a speed ranking.
The fake check measures integration and storage, not Qwen performance.

Local raw evidence is under
`artifacts/m11/qualification/20260925T204612Z-4b-development` and
`artifacts/m11/packet-2/recording-cost`. The pre-live source receipt and archive
are in `artifacts/m11/packet-2/pre-live-source.json`; the base is `fa58649`.
These local artifacts are not bundled with an installation. Saved actions and
requests support audits; temperature zero does not guarantee identical fresh
model replies. The [qualified recipe](qwen_recipe.md) reports the completed fixed model
comparison and combined GPU workload.

## Play In DevClient

From a source checkout, offer a factory when starting DevClient:

```bash
export MARL_LLM_MODEL=my-served-model
export MARL_LLM_URL=http://127.0.0.1:8000/v1
export MARL_LLM_HISTORY=2
JAX_PLATFORMS=cpu python scripts/dev/debug_renderer.py \
  --offer-system qwen=examples.llm:make_default_system
```

Choose **Qwen** for either team and keep **SharedObs** selected. Each Submit
asks the Systems for one joint turn. The browser stays usable while they work.
**Cancel System Work** abandons the pending result and keeps the last accepted
game state. A cancelled decision requires fresh methods before another turn;
Reset or choose new controllers. Cleanup never stops the model server.

Add `--offer-system words=examples.llm:make_custom_system` for the tutorial's
non-JSON replies. Only factories named by the host are offered. The browser
cannot supply a Python import, model URL or credential. Factories load only when
selected and must return fresh method instances for replacement. Both teams can
select one declaration: it loads once and keeps separate team memories.

Loading an authored map or scenario prepares fresh methods in the same worker.
The old game and source snapshot remain current until the new setup succeeds.
A failed or cancelled replacement leaves the old methods usable. Cancel pending
work before retrying a different load. Authored map/scenario loading is available
only without replay recording. While recording, controller changes and Reset use
the existing discard confirmation; Finish & Review saves the accepted prefix.

Add `--record-replay artifacts/dev_client/replays/qwen.marlbg-replay.json` to
save the game. The launcher prints the adjacent model-call directory. Its
`generation-N/session.json` records full System identities and verified saved
replay prefixes. `llm.read_calls(generation_directory)` reads the separate call
rows. A row marked `played` means the service accepted that action; it does not
by itself mean the replay reached disk. An abandoned or interrupted attempt can
remain in the call files. Save As keeps the original call directory and adds the
new verified replay path to its saved-prefix list.

A Python factory or custom hook can block indefinitely. Cancellation fences its
result immediately, but the worker cannot forcibly interrupt arbitrary Python.
The single pending slot stays occupied until that work returns and cleanup ends.
HTTP calls still use the client's deadlines. Cleanup and recording failures are
reported in the host log; the browser keeps paths and server details private.
