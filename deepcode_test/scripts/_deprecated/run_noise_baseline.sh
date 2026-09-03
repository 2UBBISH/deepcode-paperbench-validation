#!/usr/bin/env bash
# ============================================================
# 判分噪声基线:对同一份产物(rice B′,archive_bprime_frankenstein)
# 用同一裁判(DeepSeek-V4-Pro, code_only)再判一次,
# 与 2026-08-26T07-16-41 那次的 0.6047 对比,量同裁判下的漂移。
#
# 并发限到 8(上游默认 100):与 fre trial3 复现轮并行,
# 100 路会把 SiliconFlow 打限流,进而拖死复现轮。
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 安全闸:复现轮在跑时不启动(100→8 已降并发,但仍不建议同时跑)
if pgrep -f "stage_b_driver.py" >/dev/null; then
    echo "❌ 检测到 stage_b_driver.py 仍在运行(复现轮未结束)。"
    echo "   等它跑完再执行本脚本,或确认可接受 API 竞争后加 --force 重跑。"
    [ "${1:-}" != "--force" ] && exit 1
    echo "   --force 已指定,继续。"
fi

# 自助摆卷:B′ 产物(= 得 0.6047 的那一份,时间戳 08-26 14:40~15:16 已核对)
rm -rf ~/pb_submissions/rice
mkdir -p ~/pb_submissions/rice/submission
cp -r "$ROOT/archive_bprime_frankenstein/generate_code/." ~/pb_submissions/rice/submission/
echo "摆卷完成: $(find ~/pb_submissions/rice/submission -type f | wc -l) 个文件(应为 24)"

cd "$ROOT/frontier-evals/project/paperbench"
export PATH="$HOME/.local/bin:$PATH"
export PB_JUDGE_CONCURRENCY="${PB_JUDGE_CONCURRENCY:-8}"

echo "==== 噪声基线判分开始 $(date '+%F %T') · 并发=$PB_JUDGE_CONCURRENCY ===="
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
    runner.recorder=nanoeval.json_recorder:json_recorder

echo "==== 判分结束 $(date '+%F %T') ===="
G=$(ls -t runs/ | head -1)
echo "结果目录: runs/$G"
