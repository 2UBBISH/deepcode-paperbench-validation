#!/usr/bin/env bash
# Reproduce Table 2 of the paper: RoBERTa-base and T5-base at 60% sparsity.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-42}
COMMON="--sparsity 0.6 --seed $SEED --epochs 40 --distill-epochs 20"

# ---- RoBERTa-base: FT / LoRA / LoRA+Prune / Prune+Distill / APT --------------
python scripts/run.py baseline --baseline ft              --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_ft
python scripts/run.py baseline --baseline lora            --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_lora
python scripts/run.py baseline --baseline lora_prune      --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_lora_prune
python scripts/run.py baseline --baseline prune_distill   --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_prune_distill
python scripts/run.py baseline --baseline lora_prune_distill --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_lora_prune_distill
python scripts/run.py apt --task sst2 --model roberta-base $COMMON --output-dir runs/rb_sst2_apt

# MNLI + SQuAD v2 for the same model
python scripts/run.py apt --task mnli  --model roberta-base $COMMON --output-dir runs/rb_mnli_apt
python scripts/run.py apt --task squad --model roberta-base $COMMON --output-dir runs/rb_squad_apt

# ---- T5-base ----------------------------------------------------------------
python scripts/run.py baseline --baseline ft         --task sst2 --model t5-base $COMMON --output-dir runs/t5_sst2_ft
python scripts/run.py baseline --baseline lora       --task sst2 --model t5-base $COMMON --output-dir runs/t5_sst2_lora
python scripts/run.py baseline --baseline lora_prune --task sst2 --model t5-base $COMMON --output-dir runs/t5_sst2_lora_prune
python scripts/run.py apt --task sst2  --model t5-base $COMMON --output-dir runs/t5_sst2_apt
python scripts/run.py apt --task mnli  --model t5-base $COMMON --output-dir runs/t5_mnli_apt
python scripts/run.py apt --task cnn_dm --model t5-base --sparsity 0.6 --epochs 16 --distill-epochs 6 --output-dir runs/t5_cnndm_apt

python scripts/collect_results.py --glob 'runs/*/result.json'
