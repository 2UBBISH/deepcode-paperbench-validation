#!/usr/bin/env bash
# ============================================================
# ⛔ 已废弃(2026-08-26),请勿使用。原因有三:
#   1. 它吞掉 driver 的退出码(`|| true`)又不看流水线状态,崩溃后会拿**上一轮
#      的旧产物**去摆卷判分 —— E1 差点因此花 ¥37 买到一个假分数;
#   2. 它读的 /tmp/stage_b_code_dir.txt 已改为按论文分文件,这里的路径不再存在;
#   3. driver 已取消 rice 默认输入(必须显式 export STAGE_B_INPUT),本脚本没有 export。
# 替代:fre 用 run_fre_trial.sh + run_fre_grade.sh;rice 用 run_clean_e2e.sh。
# ============================================================
# 阶段 B · DeepCode 真考 rice(两步:复现 → 判分)
# 用法: bash run_stage_b.sh          # 完整模式(含参考挖掘+索引,更贵更慢)
#       bash run_stage_b.sh --fast   # 快速模式(跳过第5-7步,便宜,约省一半)
# 预估: DeepCode 复现 ¥10~40(fast 约 ¥5~15)+ 判分 ¥17~25;时长 1~2.5h
# 产物: ~/pb_submissions/rice/submission/ + runs/<新组>/rice_*/grade.json
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAST=""
[ "${1:-}" = "--fast" ] && FAST="--fast"
TS=$(date +%m%d_%H%M)
LOG_B="$ROOT/stageB_deepcode_$TS.log"

echo "==== [1/3] DeepCode 复现 rice(日志: $LOG_B)===="
cd "$ROOT/DeepCode"
.venv/bin/python "$ROOT/stage_b_driver.py" $FAST 2>&1 | tee "$LOG_B" | grep -E "Progress|Phase|✅|❌|completed|error" || true

# 从驱动脚本落盘的路径文件读取产物位置
CODE_DIR=$(cat /tmp/stage_b_code_dir.txt)
echo "==== [2/3] 摆卷: $CODE_DIR → ~/pb_submissions/rice/submission/ ===="
rm -rf ~/pb_submissions/rice
mkdir -p ~/pb_submissions/rice/submission
cp -r "$CODE_DIR"/. ~/pb_submissions/rice/submission/
ls ~/pb_submissions/rice/submission/ | head

echo "==== [3/3] PaperBench 判分(code_only · DeepSeek-V4-Pro)===="
cd "$ROOT/frontier-evals/project/paperbench"
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
echo "==== 完成。结果: runs/$G/rice_*/grade.json ===="
