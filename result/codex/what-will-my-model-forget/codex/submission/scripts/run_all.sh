#!/usr/bin/env bash
# Reproduce every table / figure of the paper that is in scope.
#
#   bash scripts/run_all.sh <output-root>
#
# These runs are the full-scale experiments of the paper (36 upstream tasks x 100
# examples, per-example fine-tuning of BART0_Large / FLAN-T5_Large / FLAN-T5_3B).
# They are meant to be executed on a GPU host; see README.md for the reduced-scale
# smoke run that can be executed on CPU.
set -euo pipefail

OUT=${1:-outputs}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

run() { echo "=== $* ==="; PYTHONPATH="$ROOT" python "$ROOT/scripts/$@"; }

# Table 1 -- forecasting example forgetting (one error fixed at a time)
run run_table1.py --model bart0_large      --tuning-mode head     --output-root "$OUT"
run run_table1.py --model bart0_large      --tuning-mode full_ft  --output-root "$OUT"
run run_table1.py --model flan_t5_large    --tuning-mode head     --output-root "$OUT"
run run_table1.py --model flan_t5_large    --tuning-mode lora     --output-root "$OUT"
run run_table1.py --model flan_t5_large    --tuning-mode full_ft  --output-root "$OUT"
run run_table1.py --model flan_t5_3b       --tuning-mode head     --output-root "$OUT"
run run_table1.py --model flan_t5_3b       --tuning-mode lora     --output-root "$OUT"

# Table 2 -- in-domain / out-of-domain generalization (BART0)
run run_table2.py --model bart0_large --tuning-mode full_ft --output-root "$OUT"

# Figure 3 -- forecasting while continually refining the LM
run run_figure3.py --model flan_t5_large --tuning-mode lora --output-root "$OUT"

# Table 3 -- model refinement with replay
run run_table3.py --model bart0_large   --tuning-mode full_ft --output-root "$OUT"
run run_table3.py --model flan_t5_large --tuning-mode lora    --output-root "$OUT"
run run_table3.py --model flan_t5_large --tuning-mode full_ft --output-root "$OUT"
run run_table3.py --model flan_t5_3b    --tuning-mode lora    --output-root "$OUT"

# Table 4 -- single errors fixed separately
run run_table4.py --model bart0_large   --tuning-mode full_ft --output-root "$OUT"
run run_table4.py --model flan_t5_large --tuning-mode lora    --output-root "$OUT"
run run_table4.py --model flan_t5_large --tuning-mode full_ft --output-root "$OUT"
run run_table4.py --model flan_t5_3b    --tuning-mode lora    --output-root "$OUT"

# Sec. 5.3 -- complexity / FLOPs
run run_complexity.py --model flan_t5_large

echo "done: results under $OUT"
