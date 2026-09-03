#!/usr/bin/env bash
# 复现进度快照(论文无关,自动认最新一轮)。用法: PAPER=rice bash trial_tick.sh
# 编排器跑完全部轮次后以退出码 9 收摊(外层 `bash trial_tick.sh || exit 0` 停表)。
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; export DEEPEVOL_ROOT="$R"
PAPER="${PAPER:-rice}"
OUT="$R/deepcode_test/$PAPER"
L=$(ls -1t "$OUT"/logs/${PAPER}_trial*_deepcode_*.log 2>/dev/null | head -1)
DRV=$(pgrep -f "stage_b_drive[r]\.py" | head -1)
ORCH=$(pgrep -f "run_all_trial[s]\.sh" | head -1)

COST=$(python3 - <<'PY' 2>/dev/null || echo "?"
import json, glob, os
c = glob.glob(os.environ['DEEPEVOL_ROOT']+'/DeepCode/deepcode_lab/tasks/paper_*/logs/llm.jsonl')
if not c: print("0.00"); raise SystemExit
d = sorted(c, key=os.path.getmtime)[-1]
ti = to = 0
for l in open(d):
    l = l.strip()
    if not l: continue
    try: x = json.loads(l)
    except Exception: continue
    ti += x.get('prompt_tokens') or 0
    to += x.get('completion_tokens') or 0
print(f"{ti/1e6*4 + to/1e6*16:.2f}")
PY
)

DONE=$(ls -1d "$HOME"/pb_submissions/$PAPER/trial*/ 2>/dev/null | wc -l)
CUR=$(basename "${L:-未开始}" | sed -E "s/${PAPER}_(trial[0-9A-Za-z_]+)_deepcode_.*/\1/")

if [ -z "$L" ]; then echo "[$PAPER] 尚未产生日志"; exit 0; fi

IDX=$(grep -ao "Analyzing file [0-9]*/[0-9]*" "$L" 2>/dev/null | tail -1)
DONE_REPO=$(ls -1 "$R"/DeepCode/deepcode_lab/tasks/paper_*/indexes/*_index.json 2>/dev/null | wc -l)
NREPO=$(find "$R"/DeepCode/deepcode_lab/tasks/paper_*/code_base -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
NPY=$(find "$R"/DeepCode/deepcode_lab/tasks/paper_*/generate_code -type f -name '*.py' 2>/dev/null | wc -l)
NFILL=$(find "$R"/DeepCode/deepcode_lab/tasks/paper_*/generate_code -type f -name '*.py' -size +0 2>/dev/null | wc -l)
NLINE=$(find "$R"/DeepCode/deepcode_lab/tasks/paper_*/generate_code -type f -name '*.py' -print0 2>/dev/null | xargs -0 cat 2>/dev/null | wc -l)

# 只数「异常签名行」,排除三类已确认无害的:
#   ① anyio cancel-scope(MCP 阶段切换老毛病,fre trial1 出现 24 次仍跑出 0.5184)
#   ② 日志格式化 TypeError(DeepCode 自带的 % 占位符缺陷)
#   ③ JSONRPCMessage 校验错(npx 启动 fetch 服务器时把 "run `npm fund`" 打进了
#      stdout,而 MCP 要求 stdout 只走 JSON-RPC。仅出现在连接阶段,连上即止)
ERRS=$(grep -aE "^[A-Za-z_.]*(Error|Exception|Timeout):" "$L" 2>/dev/null \
       | grep -avE "exit cancel scope|not all arguments converted|validation error for JSONRPCMessage" | wc -l)
ABORT=$(grep -acE "Process aborted|Progress stall|status=aborted" "$L" 2>/dev/null); ABORT=${ABORT:-0}
BAD=$(( ERRS + ABORT ))

if [ -n "$DRV" ]; then
  ET=$(ps -o etime= -p "$DRV" 2>/dev/null | tr -d ' ')
  AG=$(grep -aoE "Attached workflow LLM: agent=[A-Za-z]+" "$L" 2>/dev/null | tail -1)
  STAGE="${AG#Attached workflow LLM: agent=}"; STAGE="${STAGE:-启动中}"
  [ "$NREPO" -gt 0 ] && STAGE="$STAGE·已挖到 ${NREPO} 仓库"
  [ -n "$IDX" ] && STAGE="索引 ${IDX#Analyzing file }(仓库 ${DONE_REPO}/${NREPO})"
  grep -qa "Indexing completed successfully" "$L" 2>/dev/null && STAGE="索引完成·写码中"
  printf '[%s %s %s] %s | 已填充 %s/%s py · %s 行 | ¥%s | 已完成 %s/3 轮' \
    "$PAPER" "$CUR" "$ET" "$STAGE" "$NFILL" "$NPY" "$NLINE" "$COST" "$DONE"
  [ "$BAD" -gt 0 ] && printf ' | ⚠️ %s 条真异常/熔断' "$BAD"
  echo
elif [ -n "$ORCH" ]; then
  echo "[$PAPER] 轮次间隙(编排器在等待收尾)| 已完成 $DONE/3 轮 | ¥$COST"
else
  echo "[$PAPER 全部结束] 已完成 $DONE/3 轮"
  for d in "$HOME"/pb_submissions/$PAPER/trial*/; do
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -type f -not -path '*/.git/*'|wc -l) 文件 / $(find "$d" -name '*.py' -print0|xargs -0 cat 2>/dev/null|wc -l) 行"
  done
  tail -4 "$(ls -1t $R/deepcode_test/$PAPER/logs/${PAPER}_ledger.txt 2>/dev/null | head -1)" 2>/dev/null
  exit 9
fi
