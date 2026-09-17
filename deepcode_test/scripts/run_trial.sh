#!/usr/bin/env bash
# ============================================================
# PaperBench 基线运行 · 单轮复现器（论文无关）
# 只做「复现 + 摆卷」，**不判分** —— 判分由 run_grade.sh 统一跑。
#
# 用法: PAPER=sapg TRIAL=trial1 ENV_FILE=~/my.env nohup bash run_trial.sh > <日志> 2>&1 &
#       PAPER 决定论文（PaperBench id）；TRIAL 决定摆卷子目录名（~/pb_submissions/<PAPER>/<TRIAL>/）
#       ENV_FILE  含 PARATERA_API_KEY=... 的文件（只 source，不打印；也可改用 $DEEPCODE_HOME/credentials.json）
#       PREFLIGHT_ONLY=1  只验环境不花钱
#       DEEPCODE_HOME     默认 <仓库>/.deepcode-home（setup.sh 生成的口径配置）
#
# 口径（同 DeepEvol 复现线，两边一致）：DeepSeek-V4-Flash @ Paratera，思考关（compat.thinking=disabled，
# 回包 reasoning_tokens 必须为 0），规划与写码同一模型、无阶段覆盖，paper.md 末尾并入 addendum，
# 黑名单在 git 与 MCP 两层拦，参考挖掘 40 轮 / 下载 12 轮。
#
# 三道闸门（摆卷前必须全过）：
#   ① 口径闸：配置里的模型 / 思考开关 / 阶段覆盖 / maxTokens / 7 个 MCP
#   ② 假计划闸：planning_result_meta.json.source 必须是 generated（规划三连败后上游会伪造通用计划）
#   ③ 状态闸 + 产物归属：流水线状态 completed*，产物在本轮 tasks/ 下且 paper.md 标题核验
# ============================================================
set -euo pipefail
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deepcode_test/scripts
REPO="$(cd "$HERE/../.." && pwd)"                       # 仓库根
PAPER="${PAPER:-sapg}"
TRIAL="${TRIAL:-trial1}"
TS=$(date +%m%d_%H%M)
OUT="$REPO/runs/$PAPER"                                 # 日志、输入、任务归档、摆卷副本（不入库）
LOG="$OUT/logs/${PAPER}_${TRIAL}_deepcode_$TS.log"
TASKS="$REPO/DeepCode/deepcode_lab/tasks"
PB="$REPO/frontier-evals/project/paperbench"
CODE_DIR_FILE="/tmp/stage_b_code_dir_${PAPER}.txt"
STATUS_FILE="/tmp/stage_b_status_${PAPER}.txt"
SUB_ROOT="$HOME/pb_submissions/$PAPER"
export DEEPCODE_HOME="${DEEPCODE_HOME:-$REPO/.deepcode-home}"
# 工作区 = <cwd>/deepcode_lab（脚本在 DeepCode/ 里起 driver）。不要 export DEEPCODE_WORKSPACE：
# 上游 DeepCodeConfig 用 pydantic-settings 前缀 DEEPCODE_ 读环境变量，会把它当 workspace 配置对象解析而报错。

# key 只经环境变量进入；文件内容不回显
if [ -n "${ENV_FILE:-}" ]; then
  [ -f "$ENV_FILE" ] || { echo "❌ ENV_FILE 不存在: $ENV_FILE"; exit 1; }
  set -a; . "$ENV_FILE"; set +a
fi

# 每篇论文的身份关键词（摆卷前核验任务目录里的 paper.md 确实是这篇）与反抄袭仓库；
# 表里没有的论文按 blacklist.txt 第一条与 paper.md 首个标题自动推导。
case "$PAPER" in
  fre)  TITLE_KEY="functional reward encoding"; BLOCK_REPO="kvfrans/fre" ;;
  rice) TITLE_KEY="rice";                        BLOCK_REPO="chengzelei" ;;
  sequential-neural-score-estimation) TITLE_KEY="sequential neural"; BLOCK_REPO="jacksimons15327" ;;
  bam)  TITLE_KEY="batch and match";            BLOCK_REPO="modichirag/GSM-VI" ;;
  sapg) TITLE_KEY="sapg";                        BLOCK_REPO="jayeshs999/sapg" ;;
  *)
    [ -f "$PB/data/papers/$PAPER/paper.md" ] || { echo "❌ 未知 PAPER=$PAPER（PaperBench 里没有）"; exit 1; }
    TITLE_KEY="$(grep -m1 '^# ' "$PB/data/papers/$PAPER/paper.md" | sed 's/^# //' | tr 'A-Z' 'a-z' | awk '{print $1" "$2}')"
    BLOCK_REPO="$(grep -vE '^\s*(#|$)' "$PB/data/papers/$PAPER/blacklist.txt" | head -1 | sed 's#https://github.com/##')"
    echo "  ℹ️ $PAPER 未登记，自动推导：TITLE_KEY='$TITLE_KEY' BLOCK_REPO='$BLOCK_REPO'" ;;
esac

mkdir -p "$OUT/logs"
echo "==== [0/3] 预飞自检 · paper=$PAPER trial=$TRIAL home=$DEEPCODE_HOME $(date +%F\ %T) ===="

# ① 口径闸
DEEPCODE_EXPECT_MODEL="${DEEPCODE_EXPECT_MODEL:-DeepSeek-V4-Flash}" \
DEEPCODE_EXPECT_THINKING="${DEEPCODE_EXPECT_THINKING:-disabled}" \
python3 - <<'PY'
import json, os, sys
home = os.environ["DEEPCODE_HOME"]
c = json.load(open(os.path.join(home, "deepcode_config.json")))
a = c.get("agents", {})
want = os.environ["DEEPCODE_EXPECT_MODEL"]
d = a.get("defaults", {})
assert d.get("model") == want, f"agents.defaults.model={d.get('model')!r} != {want!r}（口径：全程 {want}）"
impl_model = (a.get("implementation") or {}).get("model")
assert impl_model in (None, "", want), f"implementation.model={impl_model!r} != {want!r} — 规划与写码必须同一模型"
assert not (a.get("planning") or {}).get("model"), "planning 存在模型覆盖，破坏“全程同模型”口径"
for ph in ("defaults", "implementation"):
    mt = (a.get(ph) or {}).get("maxTokens")
    if ph == "defaults" or mt is not None:
        assert (mt or 0) >= 32768, f"{ph}.maxTokens={mt} < 32768 — 会截断（坑8）"
conn = d.get("connection") or d.get("provider")
prof = (c.get("providers", {}).get("profiles") or {}).get(conn) or {}
# 每次调用的 max_tokens = min(agents.maxTokens, 模型目录里该模型的 maxOutputTokens)。21ebc57f 的目录把 deepseek 家族缺省
# 钳到 8192（2026-09-17 sapg trial1 实测），所以手动模型条目必须显式声明 maxOutputTokens ≥ 32768，才和 DeepEvol 线（32768）同口径。
if prof.get("modelCatalog") == "manual":
    entry = next((m for m in prof.get("manualModels") or [] if (m.get("id") if isinstance(m, dict) else m) == want), None)
    assert entry is not None, f"providers.profiles.{conn}.manualModels 里没有 {want}"
    mo = entry.get("maxOutputTokens") if isinstance(entry, dict) else None
    assert (mo or 0) >= 32768, f"manualModels[{want}].maxOutputTokens={mo}：模型目录会把每次调用钳到 8192；请声明 ≥ 32768（见 README §6）"
want_th = os.environ["DEEPCODE_EXPECT_THINKING"]
th = (prof.get("compat") or {}).get("thinking")
if want_th != "any":
    assert th == want_th, f"providers.profiles.{conn}.compat.thinking={th!r} != {want_th!r}（口径：思考关，每次请求带 thinking:{{type:disabled}}）"
need = {"code-implementation", "code-reference-indexer", "document-segmentation", "filesystem", "fetch", "github-downloader", "command-executor"}
missing = need - set((c.get("tools", {}).get("mcpServers") or {}))
assert not missing, f"缺 MCP: {missing}"
key_env = prof.get("apiKeyEnv") or ""
src = None
if key_env and os.environ.get(key_env):
    src = f"环境变量 {key_env}"
else:
    cred = os.path.join(home, "credentials.json")
    if os.path.exists(cred):
        try:
            if (json.load(open(cred)).get("connections") or {}).get(conn):
                src = "credentials.json"
        except Exception:
            pass
assert src, f"没有 key：既没设环境变量 {key_env or '(apiKeyEnv 未配置)'}，{home}/credentials.json 里也没有 connections.{conn}"
print(f"  ✅ 口径：连接={conn} 模型=全程 {want}，思考={th}，maxTokens≥32768（含目录 maxOutputTokens），MCP 7 项齐全；key 来源：{src}")
PY

BL=$(git config --global --get-regexp 'insteadof' || true)
echo "$BL" | grep -qi "$BLOCK_REPO" \
  || { echo "  ❌ $PAPER 的 git 反抄袭封锁缺失（应封锁 $BLOCK_REPO）；先 PAPERS=$PAPER bash setup.sh"; exit 1; }
echo "  ✅ $PAPER git 封锁在位（$BLOCK_REPO）"

[ -f "$PB/data/papers/$PAPER/paper.md" ] || { echo "  ❌ 找不到 $PAPER/paper.md（先 PAPERS=$PAPER bash setup.sh）"; exit 1; }
[ "$(wc -l < "$PB/data/papers/$PAPER/paper.md")" -gt 5 ] \
  || { echo "  ❌ paper.md 太短（LFS 未水合？）"; exit 1; }
echo "  ✅ $PAPER 论文资产就绪"
grep -q '^PB_STRUCTURED_PARSER_MODEL=' "$PB/.env" 2>/dev/null && echo "  ✅ 裁判二级解析模型已配（判分用）" || echo "  ⚠️ $PB/.env 未配 PB_STRUCTURED_PARSER_MODEL（本轮不判分，可先不管）"
[ -x "$REPO/DeepCode/.venv/bin/python" ] || { echo "  ❌ DeepCode/.venv 不存在（先 bash setup.sh）"; exit 1; }

if pgrep -f "stage_b_driver\.p[y]" >/dev/null; then
  echo "  ❌ 已有 driver 进程在跑"; exit 1
fi
echo "  ✅ 无残留进程"
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
[ -n "$TIMEOUT_BIN" ] || echo "  ⚠️ 没有 timeout/gtimeout（macOS 请 brew install coreutils 或用 env/bin 里的 timeout）；本轮没有 14h 硬顶"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "  🟢 PREFLIGHT_ONLY=1 → 预飞全部通过，到此为止（未启动复现、未花钱）"; exit 0
fi

echo "==== [1/3] 清场：归档全部旧任务目录 + 清本篇 stale 交接文件 ===="
mkdir -p "$OUT/task_archives" "$TASKS"
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

echo "==== [2/3] DeepCode 基线运行 $PAPER（完整模式；14h 硬顶；日志: $LOG）===="
# 输入按数据集来：PaperBench 给 agent 的是 paper.md + addendum.md（基准作者的澄清）。DeepCode 只吃一个 markdown，
# 所以把 addendum 作为末尾一节附在论文后面（标题 "Addendum"），与 DeepEvol 复现线的 intake 字节一致。
INPUT_DIR="$OUT/inputs"; mkdir -p "$INPUT_DIR"
if [ "${DEEPCODE_INPUT_ADDENDUM:-1}" = "1" ] && [ -s "$PB/data/papers/$PAPER/addendum.md" ]; then
  { cat "$PB/data/papers/$PAPER/paper.md"; printf '\n\n# Addendum\n\nClarifications provided with the paper by the benchmark authors (in scope; follow them):\n\n'; cat "$PB/data/papers/$PAPER/addendum.md"; } > "$INPUT_DIR/paper.md"
  export STAGE_B_INPUT="$INPUT_DIR/paper.md"
  echo "  📎 输入 = paper.md + addendum.md（sha256 $(shasum -a 256 "$STAGE_B_INPUT" | cut -c1-12)）"
else
  export STAGE_B_INPUT="$PB/data/papers/$PAPER/paper.md"
  echo "  📎 输入 = 仅 paper.md（DEEPCODE_INPUT_ADDENDUM=0 或无 addendum）"
fi
export STAGE_B_SLUG="$PAPER"
# 论文 §4.1 声称"web browsing 期间强制执行源码黑名单"，开源代码里没有实现；把 PaperBench 的 blacklist.txt
# 喂给 MCP 层强制执行（补丁 core/agent_runtime/tools/mcp.py）。git insteadOf 只挡 git 协议，挡不住 HTTP 抓取。
DENY=$(grep -vE '^\s*(#|$)' "$PB/data/papers/$PAPER/blacklist.txt" | paste -s -d, -)
export DEEPCODE_URL_DENYLIST="$DENY"
echo "  🚫 URL 黑名单已注入: $DEEPCODE_URL_DENYLIST"

# 抗限流：上游「standard + 1/2/4 秒三次」在供应商限流面前形同虚设（trial6 2026-08-28 白天写到 9/24 整轮报废）。
export DEEPCODE_LLM_RETRY_MODE="${DEEPCODE_LLM_RETRY_MODE:-persistent}"
export DEEPCODE_CHAT_RETRY_DELAYS="${DEEPCODE_CHAT_RETRY_DELAYS:-10,30,60,180,300}"
export DEEPCODE_PERSISTENT_MAX_DELAY="${DEEPCODE_PERSISTENT_MAX_DELAY:-900}"
export DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT="${DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT:-30}"
export DEEPCODE_OPENAI_REQUEST_TIMEOUT_S="${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S:-600}"
echo "  ♻️  抗限流: retry=$DEEPCODE_LLM_RETRY_MODE 退避=$DEEPCODE_CHAT_RETRY_DELAYS 上限=${DEEPCODE_PERSISTENT_MAX_DELAY}s 请求超时=${DEEPCODE_OPENAI_REQUEST_TIMEOUT_S}s"
# CodeRAG 预筛 / 逐文件分析 / 关系抽取的输出上限：上游 2000 / 1000 / 1500 对大仓库与推理模型必截断，静默回退全量索引。
export DEEPCODE_PREFILTER_MAX_TOKENS="${DEEPCODE_PREFILTER_MAX_TOKENS:-32000}"
export DEEPCODE_ANALYSIS_MAX_TOKENS="${DEEPCODE_ANALYSIS_MAX_TOKENS:-16000}"
export DEEPCODE_RELATIONSHIP_MAX_TOKENS="${DEEPCODE_RELATIONSHIP_MAX_TOKENS:-16000}"
echo "  🔍 索引 max_tokens: 预筛=$DEEPCODE_PREFILTER_MAX_TOKENS 分析=$DEEPCODE_ANALYSIS_MAX_TOKENS 关系=$DEEPCODE_RELATIONSHIP_MAX_TOKENS（上游 2000/1000/1500）"
# 规划单次调用限时（上游自带旋钮，默认 180s）
export DEEPCODE_CODE_ANALYZER_TIMEOUT_S="${DEEPCODE_CODE_ANALYZER_TIMEOUT_S:-600}"
# 参考挖掘 / 下载 agent：报告上限与迭代预算（上游 8192/4096 与 8/8 轮；两次真机 8 轮都不够出报告）
export DEEPCODE_REFERENCE_MAX_TOKENS="${DEEPCODE_REFERENCE_MAX_TOKENS:-32768}"
export DEEPCODE_DOWNLOAD_MAX_TOKENS="${DEEPCODE_DOWNLOAD_MAX_TOKENS:-16384}"
export DEEPCODE_REFERENCE_MAX_ITERATIONS="${DEEPCODE_REFERENCE_MAX_ITERATIONS:-40}"
export DEEPCODE_DOWNLOAD_MAX_ITERATIONS="${DEEPCODE_DOWNLOAD_MAX_ITERATIONS:-12}"
echo "  📚 挖掘 max_tokens=$DEEPCODE_REFERENCE_MAX_TOKENS/$DEEPCODE_REFERENCE_MAX_ITERATIONS 轮；下载 $DEEPCODE_DOWNLOAD_MAX_TOKENS/$DEEPCODE_DOWNLOAD_MAX_ITERATIONS 轮；规划限时 ${DEEPCODE_CODE_ANALYZER_TIMEOUT_S}s"
# 写码 stall 阈值与墙钟（上游 300s / 7200s；白天空响应期一次可达 30~50 分钟）
export DEEPCODE_STALL_THRESHOLD="${DEEPCODE_STALL_THRESHOLD:-7200}"
export DEEPCODE_MAX_WALL_SECONDS="${DEEPCODE_MAX_WALL_SECONDS:-21600}"
echo "  ⏱️  stall=${DEEPCODE_STALL_THRESHOLD}s 写码墙钟=${DEEPCODE_MAX_WALL_SECONDS}s"
echo "  🧠 思考=关（配置 compat.thinking=disabled；跑完核对 llm 日志 reasoning_tokens）"
# 实验开关（fix-①②③）必须关：①② 的提示词就是评分维度，开着跑出来的分数不是基线（README §对上游的改动）
for x in DEEPCODE_PLAN_COVERAGE_CHECK DEEPCODE_ALLOW_PLAN_EXTENSION DEEPCODE_POSTWRITE_COMPILE; do
  [ "${!x:-0}" = "1" ] && { echo "  ❌ $x=1：基线运行不允许开实验开关"; exit 1; }
done
echo "  ✅ 实验开关 fix-①②③ 全关"
# PLAN-3 第 7 / 7b 项（DeepEvol VENDOR 11，本仓库 README §5.1 A）：两个规划开关，不设 = 上游逐字节；S9 成对重跑时基线与本线都开
[ -n "${DEEPCODE_PLANNING_FANOUT:-}" ] && export DEEPCODE_PLANNING_FANOUT
[ -n "${DEEPCODE_PLANNER_CONTEXT_WINDOW:-}" ] && export DEEPCODE_PLANNER_CONTEXT_WINDOW
echo "  🧩 规划扇出=${DEEPCODE_PLANNING_FANOUT:-未设(关)} 规划器上下文窗口=${DEEPCODE_PLANNER_CONTEXT_WINDOW:-未设(上游 8 段/24k)}"

cd "$REPO/DeepCode"
set +e
if [ -n "$TIMEOUT_BIN" ]; then
  "$TIMEOUT_BIN" -k 60 50400 .venv/bin/python "$HERE/stage_b_driver.py" 2>&1 | tee "$LOG"
else
  .venv/bin/python "$HERE/stage_b_driver.py" 2>&1 | tee "$LOG"
fi
DRV=${PIPESTATUS[0]}
set -e
if [ "$DRV" -ne 0 ]; then
  if [ "$DRV" -eq 124 ]; then echo "❌ 触发 14h 硬顶，已杀"; else echo "❌ driver 退出码=$DRV"; fi
  echo "本轮不摆卷。日志: $LOG"
  exit 1
fi

STATUS=$(cat "$STATUS_FILE" 2>/dev/null || echo "missing")
case "$STATUS" in
  completed|completed_with_warnings) echo "  ✅ 流水线状态: $STATUS" ;;
  *) echo "⛔ 流水线状态=$STATUS —— 不摆卷，等人工判断是否用部分产物"; exit 2 ;;
esac

# ② 假计划闸
CODE_DIR_TMP=$(cat "$CODE_DIR_FILE" 2>/dev/null || echo "")
PLAN_META="$(dirname "$CODE_DIR_TMP")/planning_result_meta.json"
if [ -f "$PLAN_META" ]; then
  PLAN_SOURCE=$(python3 -c "import json;print(json.load(open('$PLAN_META')).get('source','unknown'))" 2>/dev/null || echo unknown)
  if [ "$PLAN_SOURCE" != "generated" ]; then
    echo "⛔ 计划来源=$PLAN_SOURCE（非 generated）—— 规划实际失败被上游包装成功，产物是照假计划写的空壳；判为废轮，不摆卷"
    exit 3
  fi
  echo "  ✅ 计划来源: generated（真实规划产物）"
else
  echo "  ⚠️ 找不到 $PLAN_META，无法核验计划来源（继续但请人工复核）"
fi

# ③ 状态闸 + 产物归属
CODE_DIR=$(cat "$CODE_DIR_FILE")
case "$CODE_DIR" in
  "$TASKS/"*) : ;;
  *) echo "❌ 产物路径不在本轮 tasks/ 下（疑似 stale）: $CODE_DIR"; exit 1 ;;
esac
[ -d "$CODE_DIR" ] || { echo "❌ 产物目录不存在: $CODE_DIR"; exit 1; }
TASK_DIR=$(dirname "$CODE_DIR")
if ! head -c 4000 "$TASK_DIR/paper.md" 2>/dev/null | grep -qi "$TITLE_KEY"; then
  echo "❌ 任务目录的 paper.md 不像 $PAPER（未匹配到 '$TITLE_KEY'）；拒绝摆卷"; exit 1
fi
NFILES=$(find "$CODE_DIR" -type f | wc -l)
echo "  产物: $CODE_DIR（$NFILES 个文件）"
[ "$NFILES" -ge 5 ] || { echo "❌ 产物文件数 <5，判为失败轮"; exit 1; }

# 口径核验：本轮所有回包的 reasoning_tokens 之和（思考关 = 0）
python3 - "$TASK_DIR" <<'PY' || true
import glob, json, os, sys
task = sys.argv[1]
files = glob.glob(os.path.join(task, "logs", "*.jsonl")) + glob.glob(os.path.join(task, "**", "llm*.jsonl"), recursive=True)
calls = reasoning = completion = 0
def dig(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "reasoning_tokens" and isinstance(v, (int, float)):
                yield int(v)
            else:
                yield from dig(v)
    elif isinstance(o, list):
        for v in o:
            yield from dig(v)
for f in sorted(set(files)):
    for line in open(f, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
        except Exception:
            continue
        r = list(dig(x))
        if not r and "completion_tokens" not in json.dumps(x)[:2000]:
            continue
        calls += 1
        reasoning += sum(r)
        c = x.get("completion_tokens") or (x.get("usage") or {}).get("completion_tokens") or 0
        completion += int(c) if isinstance(c, (int, float)) else 0
if calls:
    flag = "✅" if reasoning == 0 else "❌ 思考没关！"
    print(f"  {flag} 口径核验：{calls} 次调用，reasoning_tokens 合计 {reasoning}（completion {completion}）")
else:
    print("  ⚠️ 没找到带 usage 的 llm 日志，无法核验 reasoning_tokens（请查 DeepCode 的 llm 日志位置）")
PY

echo "==== [3/3] 摆卷 → $SUB_ROOT/$TRIAL/（不判分）===="
rm -rf "${SUB_ROOT:?}/$TRIAL"
mkdir -p "$SUB_ROOT/$TRIAL"
cp -r "$CODE_DIR"/. "$SUB_ROOT/$TRIAL/"
# 同时在 runs/ 下留一份副本供查看（权威副本仍是 ~/pb_submissions；两处都不入库）
mkdir -p "$OUT/submissions"
rm -rf "$OUT/submissions/$TRIAL"
cp -r "$CODE_DIR" "$OUT/submissions/$TRIAL"
ls "$SUB_ROOT/$TRIAL" | head
echo ""
echo "==== $TRIAL 完成 $(date +%F\ %T)。当前 $PAPER 已就绪的提交: ===="
ls "$SUB_ROOT"
echo "==== 判分请在全部轮次就绪后运行: PAPER=$PAPER bash run_grade.sh ===="
