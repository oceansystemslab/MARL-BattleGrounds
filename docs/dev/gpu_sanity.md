# Local GPU Qualification

Use this check to qualify a clean committed checkout on the maintainer's NVIDIA
GPU. Ordinary Python and browser CI checks run on CPU. The repository does not
send GitHub Actions jobs to the maintainer's RTX 5090.

GPU/JAX speed and efficiency are the performance targets. This particular script
checks CUDA execution and correctness; it is not a speed benchmark. Measure
compilation, synchronized execution, memory and transfers separately for an
implementation's declared workloads.

## GPU Efficiency Protocol

The user's 2026-09-16 rule sets future GPU environment batches to **32, 64, 128,
512 and 1024 only**. These are numbers of games running together. They do not
set team sizes, rollout lengths or matrix dimensions. External `vmap` uses its
total environment count; a scalar helper inside that batch remains supported.
Keep standalone scalar and odd-batch API edge checks on CPU. CPU simulation
speed is not a qualification target.

**Proportionate evidence, clarified on 2026-09-16:** correct, reasonably optimal
GPU/JAX code is the goal. A long benchmark campaign at every step is not. Begin
with source review, meaningful correctness checks and existing measurements.
Choose new measurements to answer a concrete question about the changed path.
Profile or broaden the workload only when results or unresolved risks justify
that work. Explain what decision an expensive check will inform. Resolve material
concerns without repeating checks that cannot change that decision.

Each implementation packet applies this protocol in proportion to its affected
workflows. The five sizes are allowed choices, not a mandatory cross-product for
every change. A full scaling comparison, when needed, uses these five sizes and
the declared rollout lengths, normally 16 and 128. A call is a chunk of play, not
necessarily a complete game; continue games across chunks and reset finished
games as the workflow requires.

1. **Freeze the comparison.** Record source and asset identities, GPU, driver,
   JAX dependencies, precision, keys, maps, rosters, method, outputs and cache
   settings. Compare the old and new paths on the same inputs. For a new
   convenience, also compare a simple, correct manual path doing the same work.
   Check results before interpreting speed. State any justified float tolerance.
2. **Use meaningful play.** Use the existing reactive controllers and ordinary
   episode limits for the main environment measurement. Include action selection,
   permitted observations and masks, stepping and required resets. Consume the
   outputs that the declared workflow needs so the compiler cannot remove that
   work. Keep fixed-action and short-episode reset stress tests separately named;
   they do not establish normal play or training throughput. Add fixed-model and
   learner measurements as those supported workflows are implemented. Do not
   present controller-only play as a learning update.
3. **Batch inside JAX.** Run one timed benchmark job at a time on an otherwise
   idle GPU, with all environments in that job batched together. Compile the
   numerical loop using the supported `jit`, batching and `lax.scan` route.
   Separate benchmark processes must not compete for the GPU during timing.
   Record other GPU use and repeat affected timing if interference invalidates it;
   do not stop someone else's processes.
4. **Separate the costs.** Measure construction, device placement, compilation
   and first execution separately from warmed execution. Finish queued device
   work before timing and wait for each timed result to finish. Use at least five
   warmed samples, report the median and spread, and retain the samples. Alternate
   old/new measurement blocks when needed to check order or thermal effects.
   Record cache hits so a cached load is not reported as a new compilation.
   This follows the [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
5. **Measure more than speed.** Report real environment transitions per second
   and time per rollout. Exclude reset calls and terminal padding from transition
   counts. Measure peak GPU memory, affected host memory, transfers and waits;
   distinguish whole-process usage, allocator peaks and compiler estimates.
   Check that changing ordinary same-shaped values reuses compiled work, while
   allowing separate compilations for different batch shapes or method structures.
   Profile affected paths to find repeated calculations, copies and allocations.
6. **Expose optional costs.** Use `metrics="priority"` for the main public-workflow
   result. Compare `none`, `full` and selected replay capture where affected.
   Keep the action workload and retained outputs matched within each before/after
   comparison. Measure writer/export time, bytes and host transfers separately
   and include them in complete saved-workflow timing when that path is in scope.
   Show that disabling optional work skips it. Do not silently omit an unsupported
   or out-of-memory workload, lower its batch size, or count it as a pass.
7. **Keep checks proportionate.** At development and packet qualification, use
   the smallest relevant subset of these five sizes that resolves the actual
   correctness and cost questions. Full scaling runs need a concrete reason.
   Do not multiply every map, mode and unrelated workload into every run.
   Reuse evidence for unchanged paths with a source-based explanation. Investigate
   regressions beyond the measured timing spread; explain and measure any added
   cost that buys a required benefit. Correctness and independent review remain
   separate gates. Speed evidence does not prove learning or sample efficiency.

Keep raw results in the existing related artifact directory and a short comparison
in the active packet. State exactly what was measured, what remains unmeasured
and the command needed to reproduce it. Report progress by finished workloads,
remaining workloads and the cost currently being investigated.

### Absolute Results and JAX Optimization

The user's further 2026-09-16 requirement is absolute efficiency in every
affected part of the workflow. Beating an older version is useful evidence, but
does not establish that the new version makes good use of the GPU. Report the
following actual numbers for each declared workload, alongside before/after
differences. Include units, measurement scope and any missing measurements.

| Measurement | Required Result |
| --- | --- |
| Throughput | Real environment transitions per second; report agent decisions separately when comparing different team sizes |
| Execution | Milliseconds per rollout and complete-workflow elapsed time, with repeat spread |
| Startup | Setup, model loading where used, compilation and first-execution seconds; number of compiled versions and cache hits |
| GPU Memory | Peak measured VRAM and live array sizes; distinguish process, allocator and compiler estimates |
| Host Memory and Transfers | Peak affected RAM, host/device bytes moved, transfer time and required waits |
| Recording and Output | Bytes per declared game/run, write/export time and the overhead of each enabled output |

A time estimate for one million transitions may be derived from measured
throughput, but label it as a projection and state whether it excludes startup
or output work. Do not present projected long-run performance as a measured run.

Review the code for waste and compare the least complicated correct alternatives.
Use a JAX/GPU profile when an important cost or bottleneck needs explanation.
Check the complete compiled loop,
effective environment batching, kernel work and launch gaps, repeated method
calls, configuration/parameter compilation reuse, host waits and device copies.
Keep numerical arrays on the GPU between steps. Avoid retaining intermediate
arrays that the workflow does not need. Consider buffer donation only when the
caller no longer needs the old arrays and the ownership contract permits reuse.
Measure its benefit and preserve public behavior. Precision changes, new compiler
settings or other techniques also need their own correctness and cost proof;
using more JAX features is not itself evidence of an improvement.

Run profiler captures separately from the main timing samples so instrumentation
does not distort the published speed comparison. GPU utilization can help explain
a bottleneck, but is not a substitute for completed work per second and memory
cost. Resolve material avoidable costs in the changed path, or explain a measured
cost that buys a required benefit, before accepting it. Keep Core's separate
approval boundary. Do not invent a throughput target or claim theoretical
optimality from a finite benchmark. Numerical release targets need explicit
workloads and supporting evidence.

### Required Interpretation in Every Closeout

Every future implementation report must interpret the tests and absolute costs
in simple English. Lead with the actual verdict for the declared goal, then give
the numbers, their practical meaning, concerns, missing evidence and next useful
check. A correctness pass does not establish fast execution; a speedup does not
establish efficient memory use. Distinguish observed cost increases from costs
proven avoidable. Do not infer a bottleneck from throughput scaling alone.

For example, explain whether setup is a one-time cost or repeated work, whether
memory includes model/learner state, and whether a timing covers real controller
play or only a reset stress test. State when evidence is too narrow to judge
absolute efficiency. Label projections, state which costs they exclude and do
not compare different simulators without matched work. Apply this interpretation
rule to every packet; reuse unchanged evidence honestly for documentation-only
work rather than running an unrelated benchmark again.

**Tooling alignment still required before the next GPU run:** the current
`scripts/dev/benchmark_evaluation.py` defaults and selected tests in
`scripts/dev/check_gpu.sh` still include older batch sizes. Update their owned
defaults and GPU test routes to follow this rule; keep CPU edge-case coverage and
perform the required shard-impact audit. Check the effective batch in each worker
and evaluator, including direct worker calls; a requested batch can be reduced by
a shorter episode schedule. Do not merely drop a correctness check
because its old fixture is small. Preserve historical commands and results with
their original scope. This documentation update does not itself change the
scripts, launch a run or establish results at the new sizes.

## Prepare the Locked Environment

Install dependencies before validation:

```bash
uv sync --locked --extra cuda13 --extra dev
```

The check uses `uv run --no-sync`, so it does not silently replace the installed
environment. It requires a working NVIDIA driver and `nvidia-smi`.

## Qualify a Commit

From a clean committed checkout, run:

```bash
scripts/dev/check_gpu.sh
```

The script rejects staged changes, tracked unstaged changes and nonignored
untracked files. Ignored private notes do not make the checkout dirty. It records
HEAD and its committed tree, then checks them again before reporting success.
It never stages or commits anything.

This pass is required before Codex pushes, opens a PR, merges or qualifies a
release. The separate [pre-commit gate](quality_gates.md) must first pass for the
frozen staged candidate. Changing candidate bytes invalidates the matching gate
result; keep the qualified source identity with the evidence.

For diagnosis while developing, use:

```bash
scripts/dev/check_gpu.sh --allow-dirty
```

A diagnostic pass does not qualify a commit for publication.

## What the Script Checks

The script forces `JAX_PLATFORMS=cuda`, disables memory preallocation and rejects
JIT-disablement, deprecated backend variables, pytest option overrides and CUDA
constraint bypasses. It fails if CUDA cannot initialize; CPU fallback cannot
satisfy this check.

JAX reports the generic platform name `gpu` for CUDA devices. The check also
resolves the concrete `cuda` backend and verifies its platform metadata and
device inventory. It multiplies two 2048-by-2048 float32 matrices with `jit`,
waits for the result and checks its shape, value and device placement.

The current focused tests cover the following behavior. Their older small-batch
fixtures need the alignment described above before the next GPU execution:

- Core stepping through `jit` and `lax.scan`.
- Scalar public reset inside external `vmap`.
- Native selected metrics across chunks and resets.
- Dynamic policy values, recurrent memory and actor separation.
- Selected in-memory replays without metric files or output directories.

The test list lives in `scripts/dev/check_gpu.sh`. These checks complement the
complete CPU correctness suite. They do not prove every GPU workload, geometry
neutrality, learned behavior or sample efficiency. Implementation packets name
any additional GPU proofs and cost comparisons they need.

## Runner Boundary

Removing a workflow does not isolate an installed self-hosted Actions runner.
That requires stopping/uninstalling its service and deregistering it from GitHub.
The qualification process described here does not depend on such a runner or an
automatically starting Actions service.
