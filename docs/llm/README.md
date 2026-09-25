# LLM Game Views And Actions

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
