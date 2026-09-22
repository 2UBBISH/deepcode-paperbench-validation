#!/usr/bin/env bash
# Full reproduction pipeline (Section 3 -> Section 6), paper-scale settings.
#
# This is the script that reproduces the paper end-to-end on a GPU machine.
# Every step writes its artifacts under ``artifacts/``.
#
#   bash run_all.sh                 # everything
#   bash run_all.sh probe vectors   # only the first steps
#
set -euo pipefail

MODEL=${MODEL:-gpt2-medium}
OUT=${OUT:-artifacts}
PY=${PY:-python}
STEPS=${@:-"probe vectors vocab pairs dpo eval analysis logitlens unalign examples figures"}

run () { echo "=== $* ==="; "$@"; }

has () { [[ " $STEPS " == *" $1 "* ]]; }

# ---------------------------------------------------------------- Section 3.1
if has probe; then
  run $PY scripts/train_probe.py --model "$MODEL" --out-dir "$OUT"
fi
if has vectors; then
  run $PY scripts/extract_toxic_vectors.py --model "$MODEL" --out-dir "$OUT"
fi
# ---------------------------------------------------------------- Section 3.2
if has vocab; then
  run $PY scripts/project_vocab.py --model "$MODEL" --out-dir "$OUT"
fi
# ---------------------------------------------------------------- Section 3.3
if has interventions; then
  run $PY scripts/run_interventions.py --model "$MODEL" --out-dir "$OUT"
fi
# ---------------------------------------------------------------- Section 4
if has pairs; then
  run $PY scripts/build_pairs.py --model "$MODEL" --out-dir "$OUT" \
      --probe-path "$OUT/probe/toxic_probe.pt" --n-pairs 24576 \
      --out "$OUT/pairs/pairs.jsonl"
fi
if has dpo; then
  run $PY scripts/train_dpo.py --model "$MODEL" --out-dir "$OUT" --pairs "$OUT/pairs/pairs.jsonl"
fi
# ---------------------------------------------------------------- Section 3.3/4 metrics
if has eval; then
  run $PY scripts/evaluate_model.py --model "$MODEL" --tag gpt2 --out-dir "$OUT"
  run $PY scripts/evaluate_model.py --model "$MODEL" --model-path "$OUT/dpo/model" \
      --tag gpt2_dpo --out-dir "$OUT"
fi
# ---------------------------------------------------------------- Section 5
if has analysis; then
  run $PY scripts/analyze_dpo.py --model "$MODEL" --dpo-model "$OUT/dpo/model" --out-dir "$OUT"
fi
if has logitlens; then
  run $PY scripts/logit_lens.py --model "$MODEL" --dpo-model "$OUT/dpo/model" --out-dir "$OUT"
fi
# ---------------------------------------------------------------- Section 6
if has unalign; then
  run $PY scripts/unalign.py --model "$MODEL" --dpo-model "$OUT/dpo/model" --out-dir "$OUT"
fi
if has examples; then
  run $PY scripts/examples.py --model "$MODEL" --dpo-model "$OUT/dpo/model" --out-dir "$OUT"
fi
if has figures; then
  run $PY scripts/make_figures.py --artifacts "$OUT"
fi

echo "done"
