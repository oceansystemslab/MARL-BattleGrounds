# Baseline Source Reuse

The recurrent MAPPO components start from Mava's reviewed learning code. BG
keeps its own environment, permitted actor inputs, action rules, evaluation,
recording and checkpoint ownership. This page records the source and the
independent numerical reference. A source match is not evidence of learning.

## Source Identity And License

The source is [Mava commit
`9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93`](https://github.com/instadeepai/Mava/tree/9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93).
Its Git tree is `f47f19aa124c460ce9dc23e63288ed765d5cdd47`. The source was
checked out at that exact commit and verified clean before extraction.

Mava uses Apache License 2.0. The fixture retains the complete license and the
original notices. The [source manifest](../../tests/fixtures/baseline_donor/source_manifest.json)
records each retained file's SHA256 and Git blob identity. Files ending in
`.txt` under that fixture are unchanged source evidence, not BG modules.
Do not reformat them or replace their expected results to hide a failed port.

| Original File | Reused Calculation Or Checked Contract |
| --- | --- |
| `mava/networks/torsos.py` | `MLPTorso` and its activation lookup. |
| `mava/networks/base.py` | `ScannedRNN`, `RecurrentActor` and `RecurrentValueNet`. |
| `mava/networks/heads.py` | `DiscreteActionHead`, including its initializer and masked logits. |
| `mava/networks/distributions.py` | `IdentityTransformation`; the reference uses the real TensorFlow Probability categorical distribution. |
| `mava/utils/multistep.py` | `calculate_gae`, including the successor-ending convention. |
| `mava/systems/ppo/anakin/rec_mappo.py` | PPO losses, group averaging, minibatch/epoch order and optimizer construction. |
| `mava/types.py`, `mava/systems/ppo/types.py` | Original observation, transition, parameter, optimizer and memory containers. |
| `mava/utils/training.py`, `mava/utils/config.py`, `mava/utils/network_utils.py` | Learning-rate, experience-count and action-head selection rules. |
| `mava/configs/default/rec_mappo.yaml` | Selects the Anakin execution settings, recurrent MAPPO settings and recurrent network. |
| `mava/configs/arch/anakin.yaml` | Sixteen environments in each of two data groups; sampled PPO evaluation. |
| `mava/configs/network/rnn.yaml`, `mava/configs/system/ppo/rec_mappo.yaml` | The network and learning settings below. |

The later PQN-VDN port is reserved for JaxMARL commit
`976aeb152cb184a5095021968bba94da96eb6394`. This reference does not verify that
later port. Mava's environment wrappers, experiment logger, evaluation loop,
checkpoint loader and distributed experiment harness are not copied into BG.

## Resolved Recurrent MAPPO Settings

Actor and critic each have a 128-wide Dense layer with ReLU, a 128-wide GRU,
and another 128-wide Dense layer with ReLU. They have separate parameters,
optimizer state and recurrent memory. There is no layer normalization in this
configuration. Torso kernel gain is square root of two. The actor head gain is
0.01 and the scalar value-head gain is 1. Dense biases start at zero. The
reference uses Flax 0.10.3's unchanged GRU defaults. Cross-version random
initialization is not claimed to produce identical values.

| Setting | Value |
| --- | --- |
| Actor And Critic Learning Rates | 0.00025 each; no decay. |
| Rollout Length | 128 transitions per environment. |
| PPO Epochs And Minibatches | Four epochs, two minibatches per epoch. |
| Discount And GAE Lambda | 0.99 and 0.95. |
| Policy And Value Clip | 0.2. |
| Entropy And Value Coefficients | 0.01 and 0.5. |
| Optimizers | Separate global-norm clip at 0.5, then Adam with epsilon 0.00001. |
| Recurrent Chunks | Full rollout; the unspecified donor setting resolves to rollout length. |
| Data Groups | Two groups; gradients are averaged before clipping and Adam. |

The two groups are parts of one learner. They are not independent runs. At
the initial BG batch of 32 environments, each group has 16 environments and
each minibatch has eight complete environment sequences. Advantage statistics
are computed separately within each group's minibatch. Time and agent axes
are not shuffled into independent recurrent samples.

The donor value loss has a factor of 0.5 before the maximum of clipped and
unclipped squared errors, followed by the configured value coefficient of 0.5.
Both factors belong to the checked calculation. Actor and critic gradient
norms are clipped separately.

## Deliberate BG Adaptations

BG supplies separate, versioned actor and training-only physical-state views.
The actor never receives critic state or critic memory. BG's encoder already
contains permitted class and self-row identity; the donor's `add_agent_id`
wrapper is not copied. The action head has 198 outputs for BG's exact native
action combinations. BG uses a small JAX categorical helper instead of adding
TensorFlow Probability to the runtime dependency set.

Recurrent memory resets at a real episode start. Death, respawn and collection
cutoffs do not reset it. Invalid padding keeps the prior memory. A real ending,
including a horizon draw, blocks value bootstrap; a collection cutoff does not.
The donor stores the reset before an observation and uses the carried next
reset during its reverse GAE scan. Tests must check the mapping to BG's
produced-transition ending flags.

BG removes inactive actors from learning statistics and removes forced-neutral
dead-actor choices from policy statistics while retaining their value learning.
The user selected equal averaging over **nonempty groups**, separately for
actor and critic. An empty group contributes neither a gradient nor a denominator
count. When every corresponding group is empty, skip that optimizer entirely,
including its counters and stored momentum. These are explicit adaptations;
the unchanged donor comparison uses all-valid samples.

## Independent CPU Reference

[The reference support module](../../tests/baseline_donor_reference.py) extracts
the named original syntax-tree bodies without rewriting their calculations.
It supplies imports, fixed configuration and ordinary-array packaging instead
of loading Mava's environment framework. It executes the original networks,
real categorical distribution, GAE, loss functions, two-group averaging and
four-epoch update. The original optimizer construction is extracted too.

The reference environment uses Python 3.12 and the donor's own locked versions:
JAX/JAXlib 0.5.3, Flax 0.10.3, Optax 0.2.4, TensorFlow Probability 0.25.0,
NumPy 1.26.4 and SciPy 1.12.0. Every installed dependency is fixed in
[reference-requirements.txt](../../tests/fixtures/baseline_donor/reference-requirements.txt).
This environment is separate from BG's Python 3.14/JAX 0.10.1 environment.
The repository does not acquire a TensorFlow Probability dependency.

The synthetic case uses four time steps, two groups of four environments,
five actors, 13 actor features and 17 critic features. It retains the real
128-wide networks and 198 actions. It includes nonzero memory, a reset inside
the sequence, both terminal and continuing final states, masked action categories and
old-policy offsets that exercise PPO clipping. It is a CPU numerical test,
not a simulation or learning run.

The archive stores fixed parameters and inputs, raw and masked logits, values,
memory, action log probabilities, advantages, targets, group losses and
gradients, one optimizer update, and the full original four-epoch result.
Stored parameters isolate the port from differences in random initialization
between library versions. The fixture metadata binds the archive, source
manifest, shapes, settings and reference package versions. Ordinary tests read
these NumPy arrays without loading any historical training library.

For larger current-stack checks, `build_same_stack_reference()` returns the
same original networks, GAE and loss bodies on BG's installed libraries. A
small test-only categorical wrapper supplies log probability and entropy. The
historical fixture checks this wrapper against the actual TensorFlow Probability
distribution before it is used as a reference for BG's larger input shapes.
It does not call the production categorical helper. Keep that bridge check;
two consumers sharing an unchecked replacement would not be independent proof.

To rebuild in a fresh temporary environment, run these commands from the
repository root. Use `--no-config` so BG's dependency constraints do not change
the independent historical environment. The output stays separate until its
changes have been reviewed.

```bash
UV_CACHE_DIR=/tmp/mava-reference-cache uv --no-config venv --python /usr/bin/python3.12 /tmp/mava-reference
UV_CACHE_DIR=/tmp/mava-reference-cache uv --no-config pip install --python /tmp/mava-reference/bin/python -r tests/fixtures/baseline_donor/reference-requirements.txt
JAX_PLATFORMS=cpu /tmp/mava-reference/bin/python tests/baseline_donor_reference.py --generate /tmp/mava-reference-output
```

The generator checks source hashes and exact package versions before running.
Compare each generated array with the checked archive; matching filenames are
not proof. Never regenerate expectations as a fix for a production mismatch.

## Evidence And Limits

The original source calculations execute successfully in the isolated CPU
environment. Repeated fresh processes produced the same 325 arrays and the
same 10,096,759-byte archive. Its SHA256 is
`94aa9f1424c124fa753935ec2c88c678bd263af189f7aca815dc6a2c6491322b`.
The current-stack bridge checked 278 forward, GAE, loss, gradient, parameter
and optimizer arrays, including all four epochs. The largest absolute
difference was 0.0000019074 in entropy. The comparison used absolute tolerance
0.000002 and relative tolerance 0.0002. These are measured agreement bounds for
the declared synthetic case, not universal numerical-error guarantees.

The production comparison in `tests/test_baseline_ppo_reference.py` also passes
on CPU. Current-stack initialization matches the donor exactly, including
parameter names, shapes and dtypes. Against the historical fixture, GAE matches
exactly; maximum absolute differences are 0.0000019074 for forward outputs,
0.0000004769 for losses and gradients, 0.0000000503 for the first parameter and
optimizer update, and 0.0000004769 after four epochs of grouped updates. These
production checks use absolute and relative tolerances of 0.000002 each and
require exact integer optimizer counters. The tighter relative bound is based
on the measured differences; the earlier bridge result keeps its original bound.

BG masking, information, episode-memory and System integration checks are
separate from these all-valid comparisons. A fixed-input donor match does not
establish training throughput, complete save/resume, sample efficiency, learned
tactics or competence. The smaller synthetic tensor shapes do not measure the
memory cost of BG's complete input encoder.

## Complete MAPPO Integration

The complete trainer reuses these verified networks and update equations.
Its learner adapter reconstructs compact permitted actor inputs only where
needed by the existing PPO update. It reads old values with the fixed rollout
critic, keeps separate critic memory and discards bootstrap-only memory advance.
Action-time categorical choices and log probabilities come from collection;
the adapter makes no second behavior-policy call or duplicate GAE calculation.

Model initialization folds tag `0x4D415050` into the run's Threefry seed. The
separate learner shuffle root uses `0x50504F55` and folds in the next accepted
update index. Empty updates consume no shuffle step. This is learner-key schema
version 1; collection's existing independent streams retain their own version.

Synchronous Orbax checkpoints replace donor checkpoint orchestration. The pinned
handler cannot store zero-element arrays; their paths/shapes/dtypes remain in
the checked schema and known code reconstructs those empty leaves on restore.
They contain no numerical bytes. Typed keys use Orbax's native handling.
This adaptation changes neither an optimizer equation nor a learned parameter.

Validation, recording, deployment loading and analysis reuse MARL-BGs owners.
The early two-member panel is provisional. Complete-workflow tests and measured
costs supplement the original numerical comparisons; actual learning quality
still needs the declared development and demonstration evidence.

## Training Compiler Policy

Built-in MAPPO passes `xla_gpu_autotune_level=0` to its outer collection and update
compilations, including collection with recording. The shared owner is
[`training._compilation`](../../src/marl_battlegrounds/training/_compilation.py).
It changes no process-wide environment or JAX setting. The raw scan and learner
functions remain usable inside callers' own compiled loops. Generic collection
keeps JAX defaults unless its caller supplies compiler options. Evaluation keeps
its existing compiler settings.

GPU autotuning times several kernel choices while compiling. Separate timing
trials can select different choices, whose floating-point calculations round
differently. Turning autotuning off fixes these compilation choices; it does
not remove every possible source of runtime variation. This distinction follows
[OpenXLA's determinism guide](https://openxla.org/xla/determinism).
JAX requires these options on the outermost compiled call, so inner numerical
helpers do not set them. See [JAX compiler controls](https://docs.jax.dev/en/latest/201/controlling-xla.html).

The fixed choice removes a restart dependency on live timing trials without a
saved compiler-cache lifecycle. A persisted autotuning cache is an alternative,
but it needs version, hardware and completeness checks. Missing entries can
otherwise trigger new timing trials. See [OpenXLA's cache rules](https://openxla.org/xla/persisted_autotuning).
The policy changes GPU execution choices, not the donor's equations, learning
settings, actor information or random-key rules. Different rounding can still
change a later sampled trajectory, so results from an earlier policy keep their
original identity.

New learner checkpoints bind this policy and the active backend, device kind,
runtime version, JAX precision, random-number and JIT settings, the profile-guided
compilation switch, and declared `XLA_FLAGS`. Resume requires
the saved identity to match before recording or log recovery. The environment
string records what was declared; changing it after JAX starts does not prove
the compiler changed. Historical descriptions, frozen actor loading and reports
remain readable; a missing old execution identity is not treated as the new one.

The first fresh-process GPU check restored all 332 array leaves exactly but
later chose different actions under ordinary compiler settings. Setting the
global autotune flag to zero produced exact continuation in two fresh child
processes on the internal RTX 5090. The final per-compilation policy then passed
the actual public trainer's fresh-process checks with global `XLA_FLAGS` absent.
From update 8 through updates 9 and 10, all 332 state leaves without recording
and 333 with recording matched bit for bit, as did actor arrays, action-time
witnesses, scientific logs, exposure and export digests. The recorded case also
matched original registrations and every CSV byte, with 32 new completed rows.
The original failure and the separate attempt rejected for changing source bytes
remain preserved.

The final B32/T128, four-epoch numerical comparison matched all 354 output leaves
exactly in initial, continued and changed-seed cases. A fixed-input compiler
comparison measured 1.17888 seconds per block for this policy versus 1.17619 for
ordinary settings, about 0.23% more. It preserved their floating-point differences
and makes no trajectory-equivalence claim. Full costs and memory limits are in
[the measured engineering checks](README.md#measured-mappo-engineering-checks).
The evidence concerns the declared workload and software stack, not equality
across devices, library upgrades, arbitrary compiler flags or all JAX programs.
