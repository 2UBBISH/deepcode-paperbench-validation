#!/usr/bin/env bash
# Reproduce Table 4 (RoBERTa-base ablations) and the GLUE part of Table 8.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--task sst2 --model roberta-base --sparsity 0.6 --epochs 40 --distill-epochs 20"

python scripts/run.py apt                                  $COMMON --output-dir runs/abl_apt
python scripts/run.py ablation --ablation wo_adaptive_pruning $COMMON --output-dir runs/abl_wo_ap
python scripts/run.py ablation --ablation wo_adaptive_tuning  $COMMON --output-dir runs/abl_wo_at
python scripts/run.py ablation --ablation wo_distillation     $COMMON --output-dir runs/abl_wo_ds
python scripts/run.py ablation --ablation wo_kurtosis         $COMMON --output-dir runs/abl_wo_kurt
python scripts/run.py ablation --ablation wo_salience         $COMMON --output-dir runs/abl_wo_sal

python scripts/collect_results.py --glob 'runs/abl_*/result.json'
