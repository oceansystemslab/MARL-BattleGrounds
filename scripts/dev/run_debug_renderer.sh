#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Historical compatibility entry point. The live product is the DevClient;
# argument parsing and exit behavior remain owned by its canonical launcher.
exec "${SCRIPT_DIR}/run_dev_client.sh" "$@"
