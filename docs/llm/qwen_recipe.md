# Qualified Local Qwen Recipe

Use **Qwen3.5-4B with dynamic FP8** as the starting recipe. It completed the
fixed software checks and scored 2 points against 9B's 1 point in the small
validation panel. This selects a usable recipe; it does not show that 4B is
usually stronger, that either model plays well, or that LLM play is efficient.

The checks ran on 25–26 September 2026 on one internal RTX 5090, shared with
unrelated training. All speed figures below include that contention. The
simulator used CPU for the development and validation games, and the same GPU
as the model for the separate 32-environment check.

## Start The Model Server

Use a separate model-serving environment. The tested installation used Python
3.12.13, vLLM 0.30.0, PyTorch 2.13 with CUDA 13.0, Transformers 5.17.0 and the
following original model weights. Dynamic FP8 quantization happens when vLLM
loads those weights; these are not pre-quantized FP8 model repositories.

| Model | Model Revision |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

Make those pinned weights available locally first. Set `QWEN_MODEL_DIR` to the
4B directory and use the serving environment's `vllm` command:

```bash
export QWEN_MODEL_DIR=/path/to/pinned/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
vllm serve "$QWEN_MODEL_DIR" \
  --host 127.0.0.1 --port 8000 --served-model-name qwen3.5-4b-fp8 \
  --language-model-only --quantization fp8 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.50 --no-enable-prefix-caching \
  --generation-config vllm --seed 0
```

The measured card was pinned by UUID
`GPU-6b11a0c4-14e8-6782-8932-1df56d599796`; use the matching card on your machine.
Wait for the server to become healthy before running a game. This command stays
in the foreground. Stop it yourself when finished. Closing an LLM System closes
its client, not this independent server. Construction makes no generation call
and downloads no weights. MARL-BGs needs no new HTTP package.

Check available GPU memory before starting beside another job. The server's
50% setting is its configured memory budget, not a limit on the entire card.
Changing versions, model content, quantization or memory settings makes a new
recipe that needs its own checks.

## Run And Inspect A Game

In the MARL-BGs environment, save this as `qwen_game.py` and run
`JAX_PLATFORMS=cpu python qwen_game.py`:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm

system = llm.make_system(
    "qwen3.5-4b-fp8", "http://127.0.0.1:8000/v1",
    model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    history_turns=0,
    failure_policy="fallback",
    records="full",
)
result = marl_bgs.evaluate(
    system, "random", maps=[0], num_episodes=2, num_envs=2,
    spawn_mode="paired", seed=19671501, max_steps=300,
    output_dir="runs/qwen", save_replays=2, chunk_size=1,
)
print(result.table("episodes"))
print(result.table("priority_metrics"))
print(llm.call_summary(result))
print(result.run_dir)
print(result.paths)
```

This example opts into the qualification's checked Stay/no-combat fallback.
The normal API default is to stop on a failed reply or request. Always report
fallbacks for both teams with outcomes. Configuration, context-limit, custom
code, simulator and recording errors still stop. Full records save exact
requests for audits; ordinary runs can use `records="light"` to save disk.

The default vLLM route resolves and sends these request settings explicitly:

| Setting | Value |
| --- | --- |
| Thinking | Off |
| Temperature / Top P / Seed | `0 / 1 / 0` |
| Top K / Min P | `-1 / 0` |
| Presence / Frequency Penalty | `0 / 0` |
| Repetition Penalty | `1` |
| Min Tokens / Ignore EOS | `0 / false` |
| Maximum Reply Tokens | `64` |
| Client Concurrency / Request Deadline / Retries | `16 / 60 seconds / 0` |

Requests use one fixed named JSON reply format. Current masks still decide
legality locally. Invalid names or combat pairs are not repaired. Inputs use
the left-spawn frame and decimal numbers that preserve each supplied float32.
Each actor receives only its allowed view and its own history. History is off
by default; `history_turns=2` retains up to two earlier living turns. The adapter adds 64 reserved reply tokens to
vLLM's wrapped input token count. Only whole old history
entries may be dropped to fit; current information is never silently cut.

For the package CLI and DevClient, use the
[default or custom factories](README.md#use-a-factory-from-the-package-command).
Set `MARL_LLM_MODEL=qwen3.5-4b-fp8` and `MARL_LLM_REVISION` to the pinned 4B
revision. Those convenience factories keep the API defaults: stop on failure
and light records. The [custom-format tutorial](custom_formats.md) changes the
prompt and then uses a non-JSON reply without replacing history or scheduling.
The same System supports authored states, custom tournaments and installed local
canonical snapshots; see the [workflow guide](README.md).

Load saved results with `marl_bgs.load_results(result.run_dir)`. Read separate
calls with `llm.read_calls(result.run_dir)`. To resume, repeat the evaluation call with the same System and schedule settings
(including `num_episodes`), using `resume_from` instead of `output_dir`. Completed durable games make no new model
requests; unfinished games restart with empty history and a new attempt ID.
Changing the declared method prevents silent reuse. Saved replay actions can
reproduce the recorded game without contacting the model. See
[recording and recovery](README.md#read-calls-and-recovery) for crash limits.

## Fixed Validation Panel

Each model played a spawn pair on map 42 against Random (seed 19671542), then a
spawn pair on map 43 against `tdm-alpha` (seed 19671543). History was off and the
horizon was 300 ticks. Settings, seeds, opponents and selection rules were fixed
before these runs. No protected test maps or scenarios shaped prompts or model
selection. All four games remain in the results, including games with fallback.

| Model | Wins / Draws / Losses | Points | Kills Per Game | Kills / Deaths | Game Lengths | Fallbacks, A / B |
| --- | --- | ---: | ---: | --- | --- | --- |
| 4B FP8 | 0 / 4 / 0 | 2 | 0.5 | 2 / 0 | 300, 300, 300, 300 | 54 / 0 |
| 9B FP8 | 0 / 2 / 2 | 1 | 0 | 0 / 25 | 300, 300, 73, 137 | 3 / 0 |

A win earns 1 point; a draw earns 0.5. Ties use kills per game, then lower
complete validation time, then 4B. The first rule selected 4B. Against Random,
4B made one kill in each draw; against ALPHA it made no kills. 9B made no kills,
lost one unit across its Random pair and lost 12 units in each ALPHA game.
These observations show limited combat success. Four games cannot establish
model superiority or a useful general win rate.

The 4B failures were 53 masked actions and one truncated reply, or 0.9003% of
its 5,998 model calls. The 9B failures were three truncated replies, or 0.0756%
of 3,968 calls. Each used one checked fallback. Both opponents had zero
fallbacks. There were no transport failures. Dead actors made no calls; forced
single choices also skipped the model. Do not infer better play from fewer calls
when a model loses earlier or has dead actors.

| Validation Cost | 4B FP8 | 9B FP8 |
| --- | ---: | ---: |
| Complete Evaluation Time, Four Games | 1,134.44 seconds | 1,253.39 seconds |
| Complete Games Per Hour, This Panel | 12.69 | 11.49 |
| Server Startup To Healthy | 31.28 seconds | 62.64 seconds |
| Model / Tokenizer Calls | 5,998 / 5,998 | 3,968 / 3,968 |
| Input / Output Tokens | 21,577,209 / 72,780 | 14,719,188 / 47,848 |
| Full Game Records And Replays | 186.76 MB | 125.96 MB |
| Sampled Owned RAM / VRAM | Incomplete measurement | 8.97 GB / 14,822 MiB |

Evaluation time includes setup, stepping, model requests and recording, but not
server startup or the later repeat check. The 4B supervisor lost its control
session after successful completion. Its server stayed idle until recovery.
The full 4,412.88-second window, including that idle time, counts against the
live budget; its RAM/VRAM sampling is incomplete. The completed game and call
records were retained and checked. No replacement run was made.

## Development, Repeated Answers And Saved Playback

Two 4B development games on map 0 against Random used seed 19671501, opposite
spawn ends, history settings 0 and 2, and 300 ticks each. Both drew without kills,
deaths, failed replies or fallbacks. They made 1,496 and 1,499 model calls and
consumed 4,459,359 and 10,536,764 input tokens. Complete evaluation took 262.39
and 529.66 seconds. These different inputs check history operation; they are
not a history-quality or controlled speed comparison. Further costs are in the
[development measurement](README.md#measured-development-check).

Each model answered the same fixed 16 saved development requests twice. All 32
replies per model were legal. All 16 pairs matched in exact text and action.
The panel includes all five actor slots and both spawn ends, but only 15 unique
request bodies; nine requests have no history, seven have two turns, and only
one offers enemy-target combat. This small observation is not a repeatability
guarantee. The installed Qwen3.5 `GDN_ATTN` backend rejects vLLM's batch-invariance
mode. Neither temperature zero nor a seed supplies that missing guarantee.
Prefix caching was off throughout; this recipe makes no claim about its speed
or answer consistency when enabled.

Both development action tapes were played through ordinary evaluation without
a server: 602 captured frames and 600 transitions matched the original numerical
replay content, including non-Stay movement. Generated run identifiers differ
by design. This checks captured replay content, not every private Core field
or reconstruction of unsaved prompts. Exact saved-action reproduction and
repeatable fresh model answers are separate claims.

## Combined Model And GPU Simulator

The selected 4B recipe also ran 32 environments for 8 ticks on development map
0 against Random, with paired spawn ends, seed 19671500, no history and full
records. The model and actual simulator arrays used the same internal GPU.
No extra live rollout was added for warm-up.

To use that simulator memory setup, set these variables **before Python starts**,
then use the ordinary evaluator's `num_envs=32` setting:

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_CLIENT_MEM_FRACTION=0.10
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
```

Use the same `system` construction from the earlier Python example, then replace
its evaluation call with:

```python
result = marl_bgs.evaluate(
    system, "random", maps=[0], num_episodes=32, num_envs=32,
    spawn_mode="paired", seed=19671500, max_steps=8,
    output_dir="runs/qwen-combined", save_replays=32, chunk_size=1,
)
print(result.table("episodes"))
print(llm.call_summary(result))
```

`chunk_size=1` means one tick per chunk; it does not reduce the 32 environments.
This runs the workload. The table below also includes separate qualification
instrumentation for timing and resource samples.

The qualification checked free memory for the 50% server budget, 10% JAX pool
and 512 MiB margin. This is a starting headroom check, not protection from another
job allocating later. The JAX pool setting does not cap every CUDA allocation.

| Measurement | Result |
| --- | ---: |
| Complete Live Window / Server Startup | 223.77 / 34.29 seconds |
| Complete Evaluation | 186.99 seconds |
| Real Transitions / Model Calls | 256 / 1,280 |
| Complete Throughput | 1.37 transitions per second |
| Eight-Tick Games Per Hour | 616.09 |
| First Tick, All 32 Environments | 29.93 seconds |
| Median Of Ticks 2–8 | 20.58 seconds |
| Setup Before First Tick | 2.75 seconds |
| Work Between And After Ticks | 9.90 seconds |
| Input / Output Tokens | 3,636,439 / 15,430 |
| Sampled Owned RAM / VRAM | 7.73 GB / 17,528 MiB |
| Sampled Whole-Card VRAM, Including Other Jobs | 23,549 MiB |
| Explicit GPU `device_get` Transfers / Time | 55.33 MB / 0.213 seconds |
| Full Game Records And Replays | 39.86 MB |
| Fallbacks, A / B | 0 / 0 |

All 32 short games drew with zero kills and deaths. Eight-tick games per hour
must not be compared with 300-tick games per hour. Tick 1 includes compilation;
later ticks still include the model, simulator and evidence. Their difference
is not a pure compilation-time measurement. The transfer count omits implicit
runtime copies. One-second memory samples can miss brief peaks; summed process
RSS can count shared pages more than once. These are contended measurements,
not isolated hardware limits.

## Prompt And Storage Cost

The real compact formatter's ten development views used 2,733–4,060 text tokens,
or 2,745–4,072 with the pinned Qwen chat wrapper. Per-view median formatting time
was 0.67–0.92 milliseconds on CPU. The full-array reference used 11,354–11,749
tokens and about 7.1–7.3 milliseconds. Exact float32 numbers remain enabled.
The measurements used development maps 0/1 and synthetic tactical positions;
no protected scenario supplied a prompt. The offline tokenizer measurement took
8.7–21.6 milliseconds for multiple counts and chat rendering, not one HTTP call.
The live records include tokenizer calls and usage; these runs did not isolate
an overall formatter or server-tokenization latency total.

Light records keep hashes and replies; full records also keep exact requests.
The fixed fake-provider 32-by-8 comparison wrote 0, 0.813 and 16.19 MB of call
records for off, light and full modes, with equal game results. It checks the
storage choice without conflating it with model speed. The 42 live games wrote
459.76 MB of full game records and replays in total. A hash verifies content;
it cannot recreate an omitted prompt.

## Evidence And Limits

The fixed live workload finished inside all original limits: **6,883.46 seconds,
14,305 model-call attempts and 55,236,163 input tokens**, against ceilings of
7,200 seconds, 20,000 attempts and 120 million tokens. Loading, tokenization,
repeat requests, idle server time and cleanup counted toward elapsed time.
No retry or extra live trial was added. All owned model servers were stopped;
other GPU jobs were left alone.

Local evidence is under `artifacts/m11/qualification/`:

- `live-budget.json` records every charged window and the unchanged limits.
- `20260925T204612Z-4b-development` holds the two development runs.
- `20260925T223616Z-4b-validation` and `20260925T235044Z-9b-validation` hold the
  paired validation runs, requests, responses and repeat summaries.
- `20260926T001744Z-4b-combined` holds device placement, eight tick timings,
  transfers, memory samples and the 32-game output.
- `saved-action-playback/proof.json` joins the two playback checks to source
  replay hashes. `model-files.json` pins local model content.
- `repeat-requests.json` has SHA-256
  `0f3035ee6924fe5be11a00cf1bf5fbc85eadf0d01242db4044596179b324eb81`.

Development used the retained pre-live source archive, based on `fa58649`, with
SHA-256 `b4f2c4140eac68621e1450181022c18aadbf4bf65318edf7f38b9e80a6109d1d`.
The source receipt is `artifacts/m11/packet-2/pre-live-source.json`. Later
workflow and DevClient changes have separate fake-server regression evidence;
these earlier live measurements must not be relabeled as runs of a later tree.
Raw artifacts are local and are not bundled with an installation.

The evidence supports a working integration, faithful allowed inputs, bounded
client work, visible fallback costs and saved-action playback. It does not
establish sample efficiency, strong tactical teamwork or manuscript-scale model
rankings. Use the short default and custom examples to build an experiment;
measure actual behavior before making those claims.
