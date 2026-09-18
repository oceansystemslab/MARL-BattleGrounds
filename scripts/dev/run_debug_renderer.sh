#!/usr/bin/env bash

# Forward the historical renderer command to the DevClient launcher.
# Usage: scripts/dev/run_debug_renderer.sh [DevClient arguments]. All arguments
# and the final process status pass through unchanged. The canonical launcher
# owns environment setup, argument parsing and server/static-renderer behavior.
# The script can be called from any working directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Historical compatibility entry point. The live product is the DevClient;
# argument parsing and exit behavior remain owned by its canonical launcher.
exec "${SCRIPT_DIR}/run_dev_client.sh" "$@"
