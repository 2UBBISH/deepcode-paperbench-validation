#!/bin/bash
# Generate stage-9 trees for a list of PaperBench papers with the vendored generator, N at a time (09-21/22 batch, re-usable).
#   GEN_ENV_DEEPSEEK=~/Documents/env/deepseek.env GEN_ENV_ALIYUN=~/Documents/env/aliyun.env RUNS=~/paper2code-runs \
#   MAXPAR=8 bash paperbench/scripts/gen_batch.sh fre rice bam ...          (no args = all 20 PaperBench papers)
# Each paper: init → run --until compute → submit to ~/pb_submissions/<paper>/deepcode. Log: $RUNS/batch.log.
# Papers whose reference repos take hours to index (adaptive-pruning/LoRA, lora-sb, masked-diffusion) are best run last.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GEN=$REPO/deepevol-deepcode/generator; PAPERS=$REPO/materials/papers; PY=$GEN/.venv/bin/python
RUNS=${RUNS:-$HOME/paper2code-runs}; MAXPAR=${MAXPAR:-8}; THINKING=${THINKING:-enabled}; POOL=${POOL:-$HOME/pb_submissions}
E1=${GEN_ENV_DEEPSEEK:-$HOME/Documents/env/deepseek.env}; E2=${GEN_ENV_ALIYUN:-$HOME/Documents/env/aliyun.env}
[ -x "$PY" ] || { echo "generator venv missing: cd $GEN && uv sync --frozen --extra agent-runtime"; exit 1; }
[ $# -gt 0 ] && LIST="$*" || LIST="fre rice bam pinn bbox bridging-data-gaps all-in-one mechanistic-understanding ftrl lbcs lca-on-the-line sapg sequential-neural-score-estimation robust-clip sample-specific-masks stay-on-topic-with-classifier-free-guidance stochastic-interpolants test-time-model-adaptation what-will-my-model-forget adaptive-pruning"
mkdir -p $RUNS; LOG=$RUNS/batch.log
running() { pgrep -f "paper2code_canary.py run --run-dir $RUNS/" | wc -l | tr -d ' '; }
one() { P=$1; cd $GEN
  [ -f $RUNS/$P/run.json ] || $PY scripts/paper2code_canary.py init --run-dir $RUNS/$P --paper-dir $PAPERS/$P --compute aliyun --figures off --planning-fanout --repair-rounds 3 --run-hours 2 \
    --model deepseek-flash --figures-model DeepSeek-V4-Flash-Vision-Exp --experiment-model DeepSeek-V4-Flash-Vision-Exp \
    --provider-base-url https://api.deepseek.com/v1 --provider-key-env DEEPSEEK_API_KEY --thinking $THINKING > $RUNS/$P.init.log 2>&1 || { echo "$P init FAILED" >> $LOG; return 1; }
  $PY scripts/paper2code_canary.py run --run-dir $RUNS/$P --until compute --env-file $E1 --env-file $E2 > $RUNS/$P/console.log 2>&1; rc=$?
  $PY scripts/paper2code_canary.py submit --run-dir $RUNS/$P --paper $P --trial deepcode --dest-root $POOL > $RUNS/$P/submit.log 2>&1 && echo "$P submitted (run rc=$rc)" >> $LOG || echo "$P run rc=$rc, submit FAILED: $(tail -1 $RUNS/$P/submit.log)" >> $LOG; }
for P in $LIST; do
  while [ $(running) -ge $MAXPAR ]; do sleep 30; done
  echo "$(date +%m-%d\ %H:%M) start $P (running $(running))" >> $LOG
  ( one $P; echo "$(date +%m-%d\ %H:%M) end $P" >> $LOG ) &
  sleep 15
done
wait; echo "$(date +%m-%d\ %H:%M) batch done" >> $LOG
