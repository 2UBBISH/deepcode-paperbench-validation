#!/usr/bin/env bash
# ============================================================
# fre 三轮串行编排器(F 系列 · 步骤 C/D/D2 → E)
#
# 为什么要有这个脚本:三轮 trial **必须串行** —— 它们共用
#   ① DeepCode 的任务目录(每轮开始会归档全部 paper_*)
#   ② /tmp/stage_b_{code_dir,status}_fre.txt 交接文件
# 并行会互相踩,产出张冠李戴。本脚本保证一轮结束才开下一轮。
#
# 用法:  nohup bash run_fre_all.sh > fre_all_console.log 2>&1 &
#        FROM=2 bash run_fre_all.sh     # 从第 2 轮开始(trial1 已完成时)
#        GRADE=0 bash run_fre_all.sh    # 只跑 trial,不自动判分
#
# 判分默认**不自动执行**(GRADE 未设=询问式:只做 DRY 报价然后停),
# 因为判分是本计划最大的单笔开销(¥45~60/份 × 4 份)。
# 要真判分请显式 GRADE=1。
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM="${FROM:-1}"
TO="${TO:-3}"
GRADE="${GRADE:-0}"
LEDGER="$ROOT/fre_all_ledger.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LEDGER"; }

cost_of_latest_task() {
  python3 - <<'PY' 2>/dev/null || echo 0
import json,glob,os
c=glob.glob('/home/deepevol/deepevol/DeepCode/deepcode_lab/tasks/paper_*/logs/llm.jsonl')
if not c: print(0); raise SystemExit
d=sorted(c,key=os.path.getmtime)[-1]
r=[json.loads(l) for l in open(d) if l.strip()]
ti=sum(x.get('prompt_tokens') or 0 for x in r); to=sum(x.get('completion_tokens') or 0 for x in r)
print(f"{ti/1e6*4+to/1e6*16:.2f}")
PY
}

log "==================== fre 三轮编排开始(轮次 $FROM..$TO)===================="

for i in $(seq "$FROM" "$TO"); do
  TRIAL="trial$i"
  SUB="$HOME/pb_submissions/fre/$TRIAL"

  # 先等干净:必须在判定"是否已完成"之前等待,否则会踩竞态 ——
  # 若上一轮正在收尾(driver 已退出但摆卷尚未完成),此刻查 $SUB 会看到空目录,
  # 从而把一个刚跑完的轮次误判为未完成并重跑。2026-08-27 实际踩到过。
  if pgrep -f "stage_b_driver\.p[y]" >/dev/null; then
    log "⏳ 检测到仍有 driver 在跑,等待其结束后再评估 $TRIAL ..."
    while pgrep -f "stage_b_driver\.p[y]" >/dev/null; do sleep 60; done
    log "   上一轮 driver 已退出,再等 60s 让摆卷/落盘收尾"
    sleep 60
  fi

  # 等干净之后再判定,此时 $SUB 的状态才是可信的
  if [ -d "$SUB" ] && [ "$(find "$SUB" -type f 2>/dev/null | wc -l)" -ge 5 ]; then
    log "⏭  $TRIAL 已有提交($(find "$SUB" -type f | wc -l) 文件),跳过"
    continue
  fi

  log "▶️  启动 $TRIAL"
  TRIAL="$TRIAL" bash "$ROOT/run_fre_trial.sh" > "$ROOT/fre_${TRIAL}_console.log" 2>&1
  RC=$?
  C=$(cost_of_latest_task)

  if [ "$RC" -eq 0 ]; then
    N=$(find "$SUB" -type f 2>/dev/null | wc -l)
    log "✅ $TRIAL 完成:$N 个文件,本轮约 ¥$C"
  else
    log "❌ $TRIAL 失败(退出码 $RC,本轮约 ¥$C)。日志: fre_${TRIAL}_console.log"
    log "   继续下一轮(单轮失败不终止整体;论文协议要的是多轮,缺一轮仍可报均值)"
  fi
done

log "==================== 三轮结束,提交清点 ===================="
ls -d "$HOME"/pb_submissions/fre/*/ 2>/dev/null | while read -r d; do
  log "   $(basename "$d"): $(find "$d" -type f -not -path '*/.git/*' | wc -l) 文件"
done

if [ "$GRADE" = "1" ]; then
  log "▶️  开始判分(GRADE=1)"
  bash "$ROOT/run_fre_grade.sh" 2>&1 | tee -a "$LEDGER"
else
  log "⏸  未自动判分(GRADE=0)。先看报价:"
  DRY=1 bash "$ROOT/run_fre_grade.sh" 2>&1 | tee -a "$LEDGER"
  log "   确认无误后执行: bash run_fre_grade.sh"
fi

log "==================== 编排结束 ===================="
