#!/usr/bin/env bash
# End-to-end smoke test: every code path with tiny networks and few iterations.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pinn.cli all --quick --outdir runs_quick "$@"
