# GPU Sanity

The reference GPU stack is validated on the self-hosted RTX 5090 runner.

The GPU CI workflow runs:

- repository tests
- formatting checks
- lint checks
- type checks
- nvidia-smi
- JAX CUDA backend smoke test

The runner is installed as a systemd service and should start automatically
after reboot.
