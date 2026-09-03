#!/usr/bin/env bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DEEPEVOL_ROOT="$R"
# ============================================================
# 多轮串行编排器(论文无关)· 不判分
#
# 为什么必须串行:各轮共用
#   ① DeepCode 的任务目录(每轮开始会归档全部 paper_*)
#   ② /tmp/stage_b_{code_dir,status}_<paper>.txt 交接文件
# 并行会互相踩,产出张冠李戴。
#
# 用法: PAPER=rice FROM=1 TO=3 nohup bash run_all_trials.sh > <日志> 2>&1 &
# ============================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="${PAPER:-rice}"
FROM="${FROM:-1}"; TO="${TO:-3}"
OUT="$HERE/../$PAPER"
LEDGER="$OUT/logs/${PAPER}_ledger.txt"
mkdir -p "$OUT/logs"
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LEDGER"; }

cost_of_latest_task() {
  python3 - <<'PY' 2>/dev/null || echo 0
import json,glob,os
c=glob.glob(os.environ['DEEPEVOL_ROOT']+'/DeepCode/deepcode_lab/tasks/paper_*/logs/llm.jsonl')
if not c: print("0.00"); raise SystemExit
d=sorted(c,key=os.path.getmtime)[-1]
ti=to=0
for l in open(d):
    l=l.strip()
    if not l: continue
    try: x=json.loads(l)
    except Exception: continue
    ti+=x.get('prompt_tokens') or 0; to+=x.get('completion_tokens') or 0
print(f"{ti/1e6*4+to/1e6*16:.2f}")
PY
}

log "========== $PAPER 串行编排开始(轮次 $FROM..$TO)=========="
for i in $(seq "$FROM" "$TO"); do
  TRIAL="trial$i"
  SUB="$HOME/pb_submissions/$PAPER/$TRIAL"

  # 先等干净:必须在判定"是否已完成"之前等待,否则会踩竞态 ——
  # 上一轮 driver 已退出但摆卷尚未完成时查 $SUB 会看到空目录,把刚跑完的轮次误判为未完成而重跑。
  if pgrep -f "stage_b_driver\.p[y]" >/dev/null; then
    log "⏳ 仍有 driver 在跑,等其结束再评估 $TRIAL ..."
    while pgrep -f "stage_b_driver\.p[y]" >/dev/null; do sleep 60; done
    log "   上一轮已退出,再等 60s 让摆卷/落盘收尾"; sleep 60
  fi

  if [ -d "$SUB" ] && [ "$(find "$SUB" -type f 2>/dev/null | wc -l)" -ge 5 ]; then
    log "⏭  $TRIAL 已有提交($(find "$SUB" -type f | wc -l) 文件),跳过"; continue
  fi

  log "▶️  启动 $TRIAL"
  PAPER="$PAPER" TRIAL="$TRIAL" bash "$HERE/run_trial.sh" > "$OUT/logs/${TRIAL}_console.log" 2>&1
  RC=$?
  C=$(cost_of_latest_task)
  if [ "$RC" -eq 0 ]; then
    log "✅ $TRIAL 完成:$(find "$SUB" -type f 2>/dev/null | wc -l) 个文件,本轮约 ¥$C"
  else
    log "❌ $TRIAL 失败(退出码 $RC,本轮约 ¥$C)。日志: $OUT/logs/${TRIAL}_console.log"
    log "   继续下一轮(单轮失败不终止整体)"
  fi
done

log "========== 结束,$PAPER 提交清点 =========="
for d in "$HOME"/pb_submissions/"$PAPER"/*/; do
  [ -d "$d" ] && log "   $(basename "$d"): $(find "$d" -type f -not -path '*/.git/*' | wc -l) 文件"
done
log "判分请运行: PAPER=$PAPER DRY=1 bash run_grade.sh  (先看报价)"
