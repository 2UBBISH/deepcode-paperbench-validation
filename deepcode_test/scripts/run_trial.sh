#!/usr/bin/env bash
# ============================================================
# PaperBench 复现 · 单轮复现器(论文无关)
# 只做「复现 + 摆卷」,**不判分** —— 判分由 run_grade.sh 统一跑,
# 避免为每轮单独付判分费,也避免为空卷付费。
#
# 用法: PAPER=rice TRIAL=trial1 nohup bash run_trial.sh > <日志> 2>&1 &
#       PAPER 默认 fre;TRIAL 决定摆卷子目录名(~/pb_submissions/<PAPER>/<TRIAL>/)
#
# 三道闸门(E1 血泪):
#   ① 每篇独立的 /tmp 交接文件(防跨论文 stale)
#   ② driver 退出码严格检查(不吞错)
#   ③ 产物必须属于本轮新任务目录 + paper.md 标题核验(防拿错论文的产物摆卷)
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deepcode_test/scripts
REPO="$(cd "$HERE/../.." && pwd)"                       # 仓库根
PAPER="${PAPER:-fre}"
TRIAL="${TRIAL:-trial1}"
TS=$(date +%m%d_%H%M)
OUT="$HERE/../$PAPER"                                   # deepcode_test/<paper>/
LOG="$OUT/logs/${PAPER}_${TRIAL}_deepcode_$TS.log"
TASKS="$REPO/DeepCode/deepcode_lab/tasks"
PB="$REPO/frontier-evals/project/paperbench"
CODE_DIR_FILE="/tmp/stage_b_code_dir_${PAPER}.txt"
STATUS_FILE="/tmp/stage_b_status_${PAPER}.txt"
SUB_ROOT="$HOME/pb_submissions/$PAPER"

# 每篇论文的身份关键词(摆卷前核验任务目录里的 paper.md 确实是这篇)与反抄袭仓库
case "$PAPER" in
  fre)  TITLE_KEY="functional reward encoding"; BLOCK_REPO="kvfrans/fre" ;;
  rice) TITLE_KEY="rice";                        BLOCK_REPO="chengzelei" ;;
  *)    echo "❌ 未知 PAPER=$PAPER,请先在本脚本登记标题关键词与封锁仓库"; exit 1 ;;
esac

mkdir -p "$OUT/logs"
echo "==== [0/3] 预飞自检 · paper=$PAPER trial=$TRIAL $(date +%F\ %T) ===="

python3 - <<'EOF'
import json, os
c = json.load(open(os.path.expanduser('~/.deepcode/deepcode_config.json')))
a = c.get('agents', {})
# 期望底座可用 DEEPCODE_EXPECT_MODEL 覆盖(如 Kimi 对照轮);默认仍是 V4-Pro
want = os.environ.get('DEEPCODE_EXPECT_MODEL', 'deepseek-ai/DeepSeek-V4-Pro')
for ph in ('defaults', 'implementation'):
    m = a.get(ph, {}).get('model', '')
    assert m == want, f'{ph}.model={m!r} != {want!r} — 本实验要求全程同底座双切'
assert not (a.get('planning') or {}).get('model'), 'planning 存在模型覆盖,破坏"全程同底座"口径'
for ph in ('defaults', 'implementation'):
    mt = a.get(ph, {}).get('maxTokens', 0)
    assert mt >= 32768, f'{ph}.maxTokens={mt} < 32768 — 推理模型会被截断(坑8)'
need = {'code-implementation', 'code-reference-indexer', 'document-segmentation',
        'filesystem', 'fetch', 'github-downloader', 'command-executor'}
missing = need - set(c.get('tools', {}).get('mcpServers', {}))
assert not missing, f'缺 MCP: {missing}'
print(f'  ✅ 模型=全程 {want}(maxTokens≥32768);MCP 7 项齐全')
EOF

BL=$(git config --global --get-regexp 'insteadof' || true)
echo "$BL" | grep -qi "$BLOCK_REPO" \
  || { echo "  ❌ $PAPER 的 git 反抄袭封锁缺失(应封锁 $BLOCK_REPO)"; exit 1; }
echo "  ✅ $PAPER git 封锁在位($BLOCK_REPO)"

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

# 只验环境不花钱:PREFLIGHT_ONLY=1 PAPER=fre bash run_trial.sh
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "  🟢 PREFLIGHT_ONLY=1 → 预飞全部通过,到此为止(未启动复现、未花钱)"; exit 0
fi

echo "==== [1/3] 清场:归档全部旧任务目录 + 清本篇 stale 交接文件 ===="
mkdir -p "$OUT/task_archives"
shopt -s nullglob
for d in "$TASKS"/paper_*; do
  [ -d "$d" ] || continue
  DEST="$OUT/task_archives/archive_task_$(basename "$d")_$TS"
  mv "$d" "$DEST"
  echo "  旧任务目录已归档 → $DEST"
done
shopt -u nullglob
LEFT=$(ls "$TASKS" 2>/dev/null | grep -c '^paper_' || true)
[ "$LEFT" -eq 0 ] || { echo "  ❌ tasks/ 仍有 $LEFT 个 paper_* 目录"; exit 1; }
rm -f "$CODE_DIR_FILE" "$STATUS_FILE"
echo "  ✅ 干净起点"

echo "==== [2/3] DeepCode 复现 $PAPER(完整模式;14h 硬顶;日志: $LOG)===="
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
# 改用 DeepCode 自带但未启用的 persistent 模式:退避最长 300s、连续同错 30 次才收手。
# 只影响「失败后等多久重试」,不改任何生成逻辑。
export DEEPCODE_LLM_RETRY_MODE="${DEEPCODE_LLM_RETRY_MODE:-persistent}"
export DEEPCODE_CHAT_RETRY_DELAYS="${DEEPCODE_CHAT_RETRY_DELAYS:-10,30,60,180,300}"
export DEEPCODE_PERSISTENT_MAX_DELAY="${DEEPCODE_PERSISTENT_MAX_DELAY:-900}"
export DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT="${DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT:-30}"
export DEEPCODE_OPENAI_REQUEST_TIMEOUT_S="${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S:-600}"
echo "  ♻️  抗限流: retry=$DEEPCODE_LLM_RETRY_MODE 退避=$DEEPCODE_CHAT_RETRY_DELAYS 上限=${DEEPCODE_PERSISTENT_MAX_DELAY}s 请求超时=${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S}s"

# CodeRAG 预筛修复:官方 max_tokens=2000 对大仓库必然截断 → json.loads 失败 →
# 静默回退「全量索引」。实测 17 文件的仓库筛选成功,151/239 文件的 100% 失败;
# google-research(8885 py)全量索引需 140h,必撞 14h 硬顶。
# 这是让论文声称的 CodeRAG 预筛真正生效,不是改变检索方法。
export DEEPCODE_PREFILTER_MAX_TOKENS="${DEEPCODE_PREFILTER_MAX_TOKENS:-32000}"
echo "  🔍 预筛 max_tokens=${DEEPCODE_PREFILTER_MAX_TOKENS}(官方默认 2000,大仓库必截断)"

# 规划单次调用限时:上游默认 180s,对 V4-Pro 的思考型输出是踩钢丝 ——
# rice 08-30 三次尝试全灭(超时/截断/超时),随后上游 coerce_text_to_minimal_plan
# 把残骸包装成通用脚手架假计划并标 completeness_score=1.0,整轮静默报废。
# 这是上游自带的环境变量旋钮(agent_orchestration_engine.py:_get_code_analyzer_timeout_s),
# 不是我方改码。
export DEEPCODE_CODE_ANALYZER_TIMEOUT_S="${DEEPCODE_CODE_ANALYZER_TIMEOUT_S:-600}"
echo "  🧠 规划单次限时=${DEEPCODE_CODE_ANALYZER_TIMEOUT_S}s(官方默认 180s,V4-Pro 思考不够用)"

# [fix-④] 参考挖掘报告 maxTokens:8192 装不下五条详版精选,截断后续写恢复只留尾段,
# 下载侧只看见 1 个 URL → 整轮语料贫瘠(trial_fx1 首跑实证)。
export DEEPCODE_REFERENCE_MAX_TOKENS="${DEEPCODE_REFERENCE_MAX_TOKENS:-32768}"
export DEEPCODE_DOWNLOAD_MAX_TOKENS="${DEEPCODE_DOWNLOAD_MAX_TOKENS:-16384}"
echo "  📚 挖掘报告 max_tokens=${DEEPCODE_REFERENCE_MAX_TOKENS} / 下载 agent=${DEEPCODE_DOWNLOAD_MAX_TOKENS}(官方 8192/4096)"

# stall 阈值:rice 计划树 35~39 文件、平均 673 行/文件,clean-slate 循环下
# V4-Pro 每文件冷启动 6.8 分钟,写码后段的连续空响应+长思考可达 30 分钟无落盘
# (trial1 2026-08-30 因此在 31/39 处被 1800s 熔断)。提到 7200s;
# 真失控仍由写码 4h 墙钟、脚本 14h 硬顶、800 迭代上限封底。
export DEEPCODE_STALL_THRESHOLD="${DEEPCODE_STALL_THRESHOLD:-7200}"
echo "  🐌 stall 阈值=${DEEPCODE_STALL_THRESHOLD}s(上游默认 300s,fre 期我方 1800s,rice 体量再放宽)"

# 写码墙钟:4h 在白天空响应期(单次卡 30~50 分钟)会掐掉健康运行 —— trial1
# 2026-08-30 写到 27/33 时剩余时间已不够。提到 6h;14h 脚本硬顶仍有余量(索引 ~2h + 写码 6h)。
export DEEPCODE_MAX_WALL_SECONDS="${DEEPCODE_MAX_WALL_SECONDS:-21600}"
echo "  ⏱️  写码墙钟=${DEEPCODE_MAX_WALL_SECONDS}s(上游 2h,fre 期我方 4h,rice 白天再放宽到 6h)"
cd "$REPO/DeepCode"
set +e
# 硬顶 14h:索引相位可长达 8h,叠加写码相位 4h 墙钟,留出余量。
timeout -k 60 50400 .venv/bin/python "$HERE/stage_b_driver.py" 2>&1 | tee "$LOG"
DRV=${PIPESTATUS[0]}
set -e
if [ "$DRV" -ne 0 ]; then
  if [ "$DRV" -eq 124 ]; then echo "❌ 触发 14h 硬顶,已杀"; else echo "❌ driver 退出码=$DRV"; fi
  echo "本轮不摆卷。日志: $LOG"
  exit 1
fi

STATUS=$(cat "$STATUS_FILE" 2>/dev/null || echo "missing")
case "$STATUS" in
  completed|completed_with_warnings) echo "  ✅ 流水线状态: $STATUS" ;;
  *) echo "⛔ 流水线状态=$STATUS —— 不摆卷,等人工判断是否用部分产物"; exit 2 ;;
esac

# 假计划闸:上游在规划三连败后会用 coerce_text_to_minimal_plan 造一个
# 通用脚手架计划(src/main.py + src/pipeline.py)并标 status=success、
# completeness_score=1.0 —— 流水线对此毫无察觉,会照着假计划写出空壳提交
# (rice 2026-08-30 实证)。摆卷前核验计划来源,非 generated 一律判废轮。
CODE_DIR_TMP=$(cat "$CODE_DIR_FILE" 2>/dev/null || echo "")
PLAN_META="$(dirname "$CODE_DIR_TMP")/planning_result_meta.json"
if [ -f "$PLAN_META" ]; then
  PLAN_SOURCE=$(python3 -c "import json;print(json.load(open('$PLAN_META')).get('source','unknown'))" 2>/dev/null || echo unknown)
  if [ "$PLAN_SOURCE" != "generated" ]; then
    echo "⛔ 计划来源=$PLAN_SOURCE(非 generated)—— 规划实际失败被上游包装成功,产物是照假计划写的空壳;判为废轮,不摆卷"
    exit 3
  fi
  echo "  ✅ 计划来源: generated(真实规划产物)"
else
  echo "  ⚠️ 找不到 $PLAN_META,无法核验计划来源(旧版任务目录?继续但请人工复核)"
fi

CODE_DIR=$(cat "$CODE_DIR_FILE")
case "$CODE_DIR" in
  "$TASKS/"*) : ;;
  *) echo "❌ 产物路径不在本轮 tasks/ 下(疑似 stale): $CODE_DIR"; exit 1 ;;
esac
[ -d "$CODE_DIR" ] || { echo "❌ 产物目录不存在: $CODE_DIR"; exit 1; }
# 产物身份核验:任务目录里的 paper.md 必须就是这篇
TASK_DIR=$(dirname "$CODE_DIR")
if ! head -c 4000 "$TASK_DIR/paper.md" 2>/dev/null | grep -qi "$TITLE_KEY"; then
  echo "❌ 任务目录的 paper.md 不像 $PAPER(未匹配到 '$TITLE_KEY');拒绝摆卷"; exit 1
fi
NFILES=$(find "$CODE_DIR" -type f | wc -l)
echo "  产物: $CODE_DIR($NFILES 个文件)"
[ "$NFILES" -ge 5 ] || { echo "❌ 产物文件数 <5,判为失败轮"; exit 1; }

echo "==== [3/3] 摆卷 → $SUB_ROOT/$TRIAL/ (不判分)===="
rm -rf "${SUB_ROOT:?}/$TRIAL"
mkdir -p "$SUB_ROOT/$TRIAL"
cp -r "$CODE_DIR"/. "$SUB_ROOT/$TRIAL/"
# 同时在 deepcode_test 下留一份副本供查看(权威副本仍是 ~/pb_submissions)
mkdir -p "$OUT/submissions"
rm -rf "$OUT/submissions/$TRIAL"
cp -r "$CODE_DIR" "$OUT/submissions/$TRIAL"
ls "$SUB_ROOT/$TRIAL" | head
echo ""
echo "==== $TRIAL 完成 $(date +%F\ %T)。当前 $PAPER 已就绪的提交: ===="
ls "$SUB_ROOT"
echo "==== 判分请在全部轮次就绪后运行: PAPER=$PAPER bash run_grade.sh ===="
