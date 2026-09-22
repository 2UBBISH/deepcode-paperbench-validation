#!/usr/bin/env bash
# End-to-end reproduction driver: pre-train every FRE variant, train the
# baselines, evaluate everything, and aggregate the paper's tables.
#
# This is the single command to reproduce the paper on a machine with the
# datasets available.  It is long-running by design (see the README's compute
# budget table); it is not meant to be run inside a sandbox.
#
#   bash scripts/run_all.sh
#
# Environment knobs:
#   SEEDS=0\\ 1\\ 2\\ 3\\ 4   training seeds (the paper uses five)
#   DEVICE=cuda            torch device
#   SKIP_BASELINES=1       skip GC-IQL / GC-BC / OPAL

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4  pre-training FRE on AntMaze =="
bash scripts/run_fre_antmaze.sh

echo "== 2/4  pre-training FRE on ExORL (walker, cheetah) =="
bash scripts/run_fre_exorl.sh

echo "== 3/4  pre-training FRE on Kitchen =="
bash scripts/run_fre_kitchen.sh

if [ "${SKIP_BASELINES:-0}" != "1" ]; then
  echo "== 4/4  training baselines (GC-IQL, GC-BC, OPAL; FB/SF via controllable_agent) =="
  bash scripts/run_baselines.sh
else
  echo "== 4/4  baselines skipped =="
fi

echo "== evaluation =="
bash scripts/evaluate_all.sh

echo
echo "Aggregated results:"
echo "  eval/table1.json                  per (domain, prior) suite scores"
echo "  eval/table1_figure5.json          Figure 5 max-normalised AntMaze scores"
echo "  eval/*.json                       per-seed raw evaluations"
