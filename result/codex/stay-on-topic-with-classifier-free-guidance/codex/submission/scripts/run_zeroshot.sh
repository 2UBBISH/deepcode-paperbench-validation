#!/usr/bin/env bash
# Section 3.1 / Table 5: zero-shot benchmark sweep over guidance strengths.
#
# Produces results/zeroshot.json, which also feeds run_flops_ancova.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_zeroshot.py \
  --models gpt2 gpt2-medium gpt2-large gpt2-xl \
           pythia-160m pythia-410m pythia-1b pythia-1.4b pythia-2.8b pythia-6.9b pythia-12b \
  --tasks arc_challenge arc_easy boolq hellaswag piqa sciq triviaqa winogrande lambada_openai \
  --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
  --batch-size 8 \
  --output results/zeroshot.json

# If lm-evaluation-harness is installed, the same table can be produced with
# the canonical task definitions instead:
#
#   python experiments/run_zeroshot.py --backend lm_eval --models gpt2-xl \
#       --gammas 1.0 1.25 1.5 --output results/zeroshot_lm_eval.json
#
#   python -m cfglm.lm_eval_adapter --model gpt2-xl --gamma 1.5 \
#       --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,sciq,triviaqa,winogrande,lambada_openai
