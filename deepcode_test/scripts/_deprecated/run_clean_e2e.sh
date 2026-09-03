#!/usr/bin/env bash
# ============================================================
# E1 · 干净端到端验证(对应 HANDOFF §3.1 / CLEAN_E2E_PLAN.md)
# 口径:新任务目录、零人工策展、一次跑完;13 项修补全保留。
# 用法: nohup bash run_clean_e2e.sh > stageE1_console.log 2>&1 &
# 对 run_stage_b.sh 修复的陷阱:
#   ① stale /tmp/stage_b_code_dir.txt + `|| true` 吞错 → 崩溃后拿旧产物判分
#   ② 判分前不看流水线状态 → 加状态闸(非 completed* 不烧判分钱)
#   ③ 产物新鲜度无校验 → 强制路径落在本轮 tasks/ 下
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS=$(date +%m%d_%H%M)
LOG="$ROOT/stageE1_deepcode_$TS.log"
TASKS="$ROOT/DeepCode/deepcode_lab/tasks"
PB="$ROOT/frontier-evals/project/paperbench"

echo "==== [0/4] 预飞自检 $(date +%F\ %T) ===="
python3 - <<'EOF'
import json
c = json.load(open('/home/deepevol/.deepcode/deepcode_config.json'))
a = c.get('agents', {})
for ph in ('defaults', 'implementation'):
    m = a.get(ph, {}).get('model', '')
    assert m == 'moonshotai/Kimi-K2.7-Code', f'{ph}.model={m!r} 不是 Kimi'
assert not (a.get('planning') or {}).get('model'), '存在 planning 模型覆盖,与"全程 Kimi"决议不符'
need = {'code-implementation', 'code-reference-indexer', 'document-segmentation',
        'filesystem', 'fetch', 'github-downloader', 'command-executor'}
missing = need - set(c.get('tools', {}).get('mcpServers', {}))
assert not missing, f'缺 MCP: {missing}'
print('  ✅ 模型=全程 Kimi;MCP 6 项齐全')
EOF
git config --global --get-regexp 'insteadof' | grep -qi 'chengzelei/rice' \
  || { echo "  ❌ git insteadOf 反抄袭封锁缺失"; exit 1; }
echo "  ✅ git 封锁在位"
grep -q '^PB_STRUCTURED_PARSER_MODEL=' "$PB/.env" \
  || { echo "  ❌ paperbench .env 缺 PB_STRUCTURED_PARSER_MODEL"; exit 1; }
echo "  ✅ 裁判二级解析模型已配"
if pgrep -f "stage_b_driver\.p[y]" >/dev/null; then
  echo "  ❌ 已有 driver 进程在跑"; exit 1
fi
echo "  ✅ 无残留进程"

echo "==== [1/4] 清场:归档旧任务目录 + 清 stale 路径 ===="
if [ -d "$TASKS/paper_e8af8afa" ]; then
  DEST="$ROOT/archive_bprime_frankenstein"
  [ -e "$DEST" ] && DEST="${DEST}_$TS"
  mv "$TASKS/paper_e8af8afa" "$DEST"
  echo "  B′ 拼装产物已归档 → $DEST"
fi
LEFT=$(ls "$TASKS" 2>/dev/null | grep -c '^paper_' || true)
[ "$LEFT" -eq 0 ] || { echo "  ❌ tasks/ 仍有 $LEFT 个 paper_* 目录,先人工处理"; exit 1; }
rm -f /tmp/stage_b_code_dir.txt /tmp/stage_b_status.txt
echo "  ✅ 干净起点(新任务目录将由流水线自建)"

echo "==== [2/4] DeepCode 复现 rice(完整模式;10h 硬顶;日志: $LOG)===="
export STAGE_B_INPUT="$PB/data/papers/rice/paper.md"
cd "$ROOT/DeepCode"
set +e
timeout -k 60 36000 .venv/bin/python "$ROOT/stage_b_driver.py" 2>&1 | tee "$LOG"
DRV=${PIPESTATUS[0]}
set -e
if [ "$DRV" -ne 0 ]; then
  if [ "$DRV" -eq 124 ]; then echo "❌ 触发 10h 硬顶,已杀"; else echo "❌ driver 退出码=$DRV"; fi
  echo "按判分闸:不自动判分。日志: $LOG"
  exit 1
fi

STATUS=$(cat /tmp/stage_b_status.txt 2>/dev/null || echo "missing")
case "$STATUS" in
  completed|completed_with_warnings)
    echo "  ✅ 流水线状态: $STATUS" ;;
  *)
    echo "⛔ 流水线状态=$STATUS —— 按判分闸停止,不烧判分钱。"
    echo "   如决定判部分产物,拿到口令后人工执行本脚本 [3/4][4/4] 两段。"
    exit 2 ;;
esac

CODE_DIR=$(cat /tmp/stage_b_code_dir.txt)
case "$CODE_DIR" in
  "$TASKS/"*) : ;;
  *) echo "❌ 产物路径不在本轮 tasks/ 下(疑似 stale): $CODE_DIR"; exit 1 ;;
esac
[ -d "$CODE_DIR" ] || { echo "❌ 产物目录不存在: $CODE_DIR"; exit 1; }
echo "  产物: $CODE_DIR($(find "$CODE_DIR" -type f | wc -l) 个文件)"

echo "==== [3/4] 摆卷 → ~/pb_submissions/rice/submission/ ===="
rm -rf ~/pb_submissions/rice
mkdir -p ~/pb_submissions/rice/submission
cp -r "$CODE_DIR"/. ~/pb_submissions/rice/submission/
ls ~/pb_submissions/rice/submission/ | head

echo "==== [4/4] 判分(code_only · DeepSeek-V4-Pro 裁判恒定)$(date +%F\ %T) ===="
cd "$PB"
export PATH="$HOME/.local/bin:$PATH"
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='deepseek-ai/DeepSeek-V4-Pro' \
    paperbench.judge.code_only=True \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder 2>&1 | tail -15

G=$(ls -t runs/ | head -1)
echo "==== E1 完成 $(date +%F\ %T)。grade: $PB/runs/$G/rice_*/grade.json ===="
echo "==== 记得回填 CLEAN_E2E_PLAN.md §4 台账与 §5 结果登记 ===="
