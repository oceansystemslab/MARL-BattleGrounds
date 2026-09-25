# Use Your Own LLM Format

Supply ordinary Python functions to `llm.make_system`. You can change the prompt,
the reply syntax, or both. The System still owns actor routing, optional history,
request limits, frame conversion and native action checks.

The runnable source is [examples/llm.py](../../examples/llm.py). Its three modes
use the same evaluator and output files. Start a compatible vLLM server first;
the example does not download weights or start a server.

## Change The Prompt

A prompt function receives `(actor, masks, history)` and returns a string. The
actor and masks describe one permitted decision in the chosen frame, left by
default. History contains only that actor's retained earlier turns. The arrays
are read-only. No game ID, private simulator state or another actor's private
view is supplied.

The example's `prompt_with_note` adds a short introduction and clearly labels
earlier turns. It reuses `format_actor_view`, so it does not copy field names or
formatting rules. These tutorial functions use the default left frame. If you
choose `frame="world"`, pass that same frame to their formatter calls. The
example keeps the default JSON parser:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm
from examples.llm import prompt_with_note

with llm.Client("http://127.0.0.1:8000/v1") as client:
    system = llm.make_system(
        "my-served-model", client=client, history_turns=2,
        prompt_builder=prompt_with_note,
        custom_version="tutorial-v1", custom_settings={"format": "note"},
    )
    result = marl_bgs.evaluate(
        system, "random", num_episodes=2, maps=[0], num_envs=2,
        output_dir="runs/llm-note",
    )

print(result.table("episodes"))
```

Your own function can arrange the permitted fields differently or use a smaller
selection. That is part of your method. It cannot grant itself more information.
Keep it pure: token fitting may call it again with fewer earlier entries.

## Change The Reply

A parser receives `(reply, actor, masks, history)` and returns `ActorAction`.
The context is the exact request that produced the reply. Raise
`llm.ReplyFormatError` for expected bad text. Other bugs should propagate.

The example's `prompt_for_words` asks for:

```text
north enemy_2_ultimate
```

Its `parse_words` splits those two names and reuses `parse_action_reply` to map
them to native categories. It does not copy Core's action-name table. The System
checks the returned action again, converts movement back to the world frame once,
and checks the original world masks. Target IDs never change with the frame.

`format_actor_view(..., reply_instruction="...")` changes the final instruction
without changing the facts. `reply_instruction=None` omits it in an earlier
history view. This prevents old JSON instructions from fighting a custom parser.
You can also write the whole prompt yourself.

```bash
JAX_PLATFORMS=cpu python examples/llm.py --model my-served-model \
  --format words --history-turns 2 --output-dir runs/llm-words
```

A custom parser turns off the default provider JSON restriction. You may supply
an explicit compatible `response_format`; it must match what your parser accepts.
The shared action checker applies either way. It checks the full target/Ultimate
pair, not just each separate category.

## Inspect The Same Game Records

`result.table("episodes")` gives outcomes, scores and game lengths.
`result.table("priority_metrics")` gives the ordinary default behavior measures.
`result.paths` names the saved files. Choose replays through the evaluator's
normal `save_replays` or `replay_episodes` settings. These game records are separate
from provider requests and replies. Custom formats use the same call records:

```python
print(llm.call_summary(result))
for call in llm.read_calls(result.run_dir):
    print(call["outcome"], call.get("reply"), call.get("world_action"))
```

Add `records="full"` to retain exact custom request text. The default light mode
keeps its hash and reply. See [records and recovery](README.md#read-calls-and-recovery)
for saved-game boundaries and interrupted attempts.

Declare a `custom_version` and frozen JSON `custom_settings` for a recorded
method. Existing callable evidence also identifies ordinary Python function
code. Changing a declared setting, generation setting or identifiable hook
changes the method identity. A version is a declaration, not proof that arbitrary
Python code obeyed it. Mutable globals, closure values and an external server's
loaded weights cannot be discovered reliably from a function's name or bytecode.
Keep those fixed and declare all answer-relevant settings for reproducible work.

For a custom token counter, return `(input_token_count, context_limit)` for the
complete supplied request, including its chat wrapper. Both values must be
integers; the count is nonnegative and the limit is positive. The adapter reserves
`max_tokens`, removes oldest whole entries when necessary, and checks the counted
input against generation usage. Changing the counter does not change the chosen
server's sampling or thinking settings. A custom counter also belongs to the
method's declared identity.
