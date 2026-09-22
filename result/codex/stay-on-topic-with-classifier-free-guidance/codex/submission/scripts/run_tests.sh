#!/usr/bin/env bash
# Offline unit tests + end-to-end smoke test (no network, CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest tests -q
python3 experiments/smoke_test.py
