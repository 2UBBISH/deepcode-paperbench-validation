#!/usr/bin/env bash
# ============================================================
# fre 三方验证 · 单轮复现器(F 系列 · 步骤 C/D)
# 只做「复现 + 摆卷」,**不判分** —— 判分由 run_fre_grade.sh 统一跑一次,
# 避免为每轮单独付判分费,也避免为空卷付费。
#
# 用法: TRIAL=trial1 nohup bash run_fre_trial.sh > fre_trial1_console.log 2>&1 &
#       TRIAL 决定摆卷子目录名(~/pb_submissions/fre/<TRIAL>/),多轮不互相覆盖。
#
# 继承 run_clean_e2e.sh 的三道闸门(E1 血泪):
#   ① 每篇独立的 /tmp 交接文件(防跨论文 stale)
#   ② driver 退出码严格检查(不吞错)
#   ③ 产物必须属于本轮新任务目录(防拿旧产物摆卷)
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER=fre
TRIAL="${TRIAL:-trial1}"
TS=$(date +%m%d_%H%M)
LOG="$ROOT/fre_${TRIAL}_deepcode_$TS.log"
TASKS="$ROOT/DeepCode/deepcode_lab/tasks"
PB="$ROOT/frontier-evals/project/paperbench"
CODE_DIR_FILE="/tmp/stage_b_code_dir_${PAPER}.txt"
STATUS_FILE="/tmp/stage_b_status_${PAPER}.txt"
SUB_ROOT="$HOME/pb_submissions/$PAPER"

echo "==== [0/3] 预飞自检 · paper=$PAPER trial=$TRIAL $(date +%F\ %T) ===="

python3 - <<'EOF'
import json
c = json.load(open('/home/deepevol/.deepcode/deepcode_config.json'))
a = c.get('agents', {})
want = 'deepseek-ai/DeepSeek-V4-Pro'
for ph in ('defaults', 'implementation'):
    m = a.get(ph, {}).get('model', '')
    assert m == want, f'{ph}.model={m!r} != {want!r} — fre 轮要求全程 V4-Pro 双切'
assert not (a.get('planning') or {}).get('model'), 'planning 存在模型覆盖,破坏"全程同底座"口径'
for ph in ('defaults', 'implementation'):
    mt = a.get(ph, {}).get('maxTokens', 0)
    assert mt >= 32768, f'{ph}.maxTokens={mt} < 32768 — 推理模型会被截断(坑8)'
need = {'code-implementation', 'code-reference-indexer', 'document-segmentation',
        'filesystem', 'fetch', 'github-downloader', 'command-executor'}
missing = need - set(c.get('tools', {}).get('mcpServers', {}))
assert not missing, f'缺 MCP: {missing}'
print('  ✅ 模型=全程 DeepSeek-V4-Pro(32768);MCP 7 项齐全')
EOF

BL=$(git config --global --get-regexp 'insteadof' || true)
echo "$BL" | grep -qi 'kvfrans/fre' \
  || { echo "  ❌ fre 的 git 反抄袭封锁缺失"; exit 1; }
echo "$BL" | grep -qi 'chengzelei' \
  || echo "  ⚠️ rice 封锁不在位(本轮不影响)"
echo "  ✅ fre git 封锁在位"

grep -q '^PB_STRUCTURED_PARSER_MODEL=' "$PB/.env" \
  || { echo "  ❌ paperbench .env 缺 PB_STRUCTURED_PARSER_MODEL"; exit 1; }
echo "  ✅ 裁判二级解析模型已配"

[ -f "$PB/data/papers/$PAPER/paper.md" ] || { echo "  ❌ 找不到 $PAPER/paper.md"; exit 1; }
[ "$(wc -l < "$PB/data/papers/$PAPER/paper.md")" -gt 5 ] \
  || { echo "  ❌ paper.md 太短(LFS 未水合?)"; exit 1; }
echo "  ✅ $PAPER 论文资产就绪"

if pgrep -f "stage_b_driver\.p[y]" >/dev/null; then
  echo "  ❌ 已有 driver 进程在跑"; exit 1
fi
echo "  ✅ 无残留进程"

echo "==== [1/3] 清场:归档全部旧任务目录 + 清本篇 stale 交接文件 ===="
shopt -s nullglob
for d in "$TASKS"/paper_*; do
  [ -d "$d" ] || continue
  DEST="$ROOT/archive_task_$(basename "$d")_$TS"
  mv "$d" "$DEST"
  echo "  旧任务目录已归档 → $DEST"
done
shopt -u nullglob
LEFT=$(ls "$TASKS" 2>/dev/null | grep -c '^paper_' || true)
[ "$LEFT" -eq 0 ] || { echo "  ❌ tasks/ 仍有 $LEFT 个 paper_* 目录"; exit 1; }
rm -f "$CODE_DIR_FILE" "$STATUS_FILE"
echo "  ✅ 干净起点"

echo "==== [2/3] DeepCode 复现 $PAPER(完整模式;10h 硬顶;日志: $LOG)===="
export STAGE_B_INPUT="$PB/data/papers/$PAPER/paper.md"
export STAGE_B_SLUG="$PAPER"
# 论文 §4.1 声称"web browsing 期间强制执行源码黑名单",但开源代码里没有任何实现。
# 这里把 PaperBench 自己的 blacklist.txt 喂给 MCP 层强制执行 —— 是补齐论文协议,
# 不是额外加料。git insteadOf 只挡 git 协议,挡不住 HTTP 抓取(trial 1 实证)。
DENY=$(grep -vE '^\s*(#|$)' "$PB/data/papers/$PAPER/blacklist.txt" | paste -sd,)
export DEEPCODE_URL_DENYLIST="$DENY"
echo "  🚫 URL 黑名单已注入: $DEEPCODE_URL_DENYLIST"

# 抗限流:官方默认「standard + 1/2/4 秒三次重试」在供应商侧限流面前形同虚设 ——
# trial6(2026-08-28)白天写码到 9/24 时连吃三次 180s 请求超时,整轮报废(¥19 白花)。
# 改用 DeepCode 自带但未启用的 persistent 模式:退避最长 300s、连续同错 30 次才收手,
# 相当于在高峰期一直等到通为止。只影响「失败后等多久重试」,不改任何生成逻辑。
export DEEPCODE_LLM_RETRY_MODE="${DEEPCODE_LLM_RETRY_MODE:-persistent}"
export DEEPCODE_CHAT_RETRY_DELAYS="${DEEPCODE_CHAT_RETRY_DELAYS:-5,15,30,60,120}"
export DEEPCODE_PERSISTENT_MAX_DELAY="${DEEPCODE_PERSISTENT_MAX_DELAY:-300}"
export DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT="${DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT:-30}"
export DEEPCODE_OPENAI_REQUEST_TIMEOUT_S="${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S:-600}"
echo "  ♻️  抗限流: retry=$DEEPCODE_LLM_RETRY_MODE 退避=$DEEPCODE_CHAT_RETRY_DELAYS 上限=${DEEPCODE_PERSISTENT_MAX_DELAY}s 请求超时=${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S}s"
cd "$ROOT/DeepCode"
set +e
# 硬顶 14h(原 10h):索引相位熔断线放宽到 6h 后,若再叠加写码相位的 4h 墙钟,
# 10h 会在写码中途把进程砍掉、白费索引开销。14h 留出余量。
timeout -k 60 50400 .venv/bin/python "$ROOT/stage_b_driver.py" 2>&1 | tee "$LOG"
DRV=${PIPESTATUS[0]}
set -e
if [ "$DRV" -ne 0 ]; then
  if [ "$DRV" -eq 124 ]; then echo "❌ 触发 10h 硬顶,已杀"; else echo "❌ driver 退出码=$DRV"; fi
  echo "本轮不摆卷。日志: $LOG"
  exit 1
fi

STATUS=$(cat "$STATUS_FILE" 2>/dev/null || echo "missing")
case "$STATUS" in
  completed|completed_with_warnings) echo "  ✅ 流水线状态: $STATUS" ;;
  *) echo "⛔ 流水线状态=$STATUS —— 不摆卷,等人工判断是否用部分产物"; exit 2 ;;
esac

CODE_DIR=$(cat "$CODE_DIR_FILE")
case "$CODE_DIR" in
  "$TASKS/"*) : ;;
  *) echo "❌ 产物路径不在本轮 tasks/ 下(疑似 stale): $CODE_DIR"; exit 1 ;;
esac
[ -d "$CODE_DIR" ] || { echo "❌ 产物目录不存在: $CODE_DIR"; exit 1; }
# 产物身份核验:任务目录里的 paper.md 必须就是 fre 那篇
TASK_DIR=$(dirname "$CODE_DIR")
if ! head -c 4000 "$TASK_DIR/paper.md" 2>/dev/null | grep -qi "functional reward encoding"; then
  echo "❌ 任务目录的 paper.md 不像 fre(未匹配到标题关键词);拒绝摆卷"; exit 1
fi
NFILES=$(find "$CODE_DIR" -type f | wc -l)
echo "  产物: $CODE_DIR($NFILES 个文件)"
[ "$NFILES" -ge 5 ] || { echo "❌ 产物文件数 <5,判为失败轮"; exit 1; }

echo "==== [3/3] 摆卷 → $SUB_ROOT/$TRIAL/ (不判分)===="
rm -rf "${SUB_ROOT:?}/$TRIAL"
mkdir -p "$SUB_ROOT/$TRIAL"
cp -r "$CODE_DIR"/. "$SUB_ROOT/$TRIAL/"
ls "$SUB_ROOT/$TRIAL" | head
echo ""
echo "==== $TRIAL 完成 $(date +%F\ %T)。当前已就绪的提交: ===="
ls "$SUB_ROOT"
echo "==== 判分请在全部轮次就绪后运行: bash run_fre_grade.sh ===="
