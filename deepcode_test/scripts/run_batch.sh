#!/usr/bin/env bash
# The 2026-09-19 batch in one command: for each paper, the DeepCode baseline run (run_trial.sh, sequential — its
# task directory and /tmp state files allow one at a time) and the two bare arms (run_bare.sh codex / claude, run in
# parallel with the baseline, each on its own proxy port). Everything ends in ~/pb_submissions/<paper>/{trial1,codex1,claude1}.
#
#   nohup bash run_batch.sh > runs/batch_0919.log 2>&1 &
#   PAPERS="sapg pinn" ARMS="baseline codex" bash run_batch.sh          # subsets
#
# Needs: ~/Documents/env/deepseek.env with DEEPSEEK_API_KEY=… (the baseline's key; the CLIs use their own cc-switch keys),
# the CLIs installed, PAPERS hydrated (setup.sh), deepcode_config on the deepseek profile (DEEPCODE_CONNECTION=deepseek).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
PAPERS="${PAPERS:-sapg pinn adaptive-pruning self-expansion test-time-model-adaptation}"   # robust-clip dropped: its official paper.md lacks §2–§3 (check_paper_md.py)
ARMS="${ARMS:-baseline codex claude}"
ENV_FILE="${ENV_FILE:-$HOME/Documents/env/deepseek.env}"
MODEL="${DEEPCODE_EXPECT_MODEL:-deepseek-flash}"
LEDGER="$REPO/runs/batch_$(date +%m%d_%H%M).txt"; mkdir -p "$REPO/runs"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LEDGER"; }
has(){ case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
has baseline && { [ -f "$ENV_FILE" ] || { echo "❌ $ENV_FILE missing (DEEPSEEK_API_KEY=… for the baseline)"; exit 1; }; }

baseline_chain() {  # sequential over papers
  for p in $PAPERS; do
    if [ -d "$HOME/pb_submissions/$p/trial1" ]; then log "baseline $p: trial1 already in the pool, skipping"; continue; fi
    log "baseline $p: start"
    PAPER="$p" TRIAL=trial1 ENV_FILE="$ENV_FILE" DEEPCODE_EXPECT_MODEL="$MODEL" \
      bash "$HERE/run_trial.sh" > "$REPO/runs/${p}_trial1_batch.log" 2>&1
    log "baseline $p: exit=$? ($(ls "$HOME/pb_submissions/$p" 2>/dev/null | tr '\n' ' '))"
  done
}
bare_chain() {  # per paper: codex and claude in parallel, then the next paper
  for p in $PAPERS; do
    pids=()
    for arm in codex claude; do
      has "$arm" || continue
      if ls -d "$HOME/pb_submissions/$p/${arm}"[0-9]* >/dev/null 2>&1; then log "$arm $p: already in the pool, skipping"; continue; fi
      log "$arm $p: start"
      ( bash "$HERE/../bare/run_bare.sh" "$arm" "$p" > "$REPO/runs/${p}_${arm}_batch.log" 2>&1; echo "$?" > "$REPO/runs/${p}_${arm}_batch.exit" ) &
      pids+=($!)
    done
    for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done
    for arm in codex claude; do has "$arm" && [ -f "$REPO/runs/${p}_${arm}_batch.exit" ] && log "$arm $p: exit=$(cat "$REPO/runs/${p}_${arm}_batch.exit") $(grep -h 'CALIBER_' "$HOME/Documents/env/bare-0919/$p-$arm/AUDIT.txt" 2>/dev/null | head -1)"; done
  done
}
log "batch start: papers [$PAPERS] arms [$ARMS] model $MODEL"
has baseline && { baseline_chain & BPID=$!; }
{ has codex || has claude; } && { bare_chain & CPID=$!; }
[ -n "${BPID:-}" ] && wait "$BPID"; [ -n "${CPID:-}" ] && wait "$CPID"
log "batch done. pool:"; for p in $PAPERS; do log "  $p: $(ls "$HOME/pb_submissions/$p" 2>/dev/null | tr '\n' ' ')"; done
log "grade: for p in $PAPERS; do PAPER=\$p PB_JUDGE_MODEL=DeepSeek-V4-Flash bash deepcode_test/scripts/run_grade.sh; done"
