#!/usr/bin/env bash
# JudgeEval on the rice/0 example (author's official repo, 178 Code-Dev leaves with human labels): the judge's accuracy
# under the current run_grade.sh caliber. Same provider / model / whole-tree / thinking switches as run_grade.sh.
#   PB_JUDGE_THINKING=off bash paperbench/scripts/run_judge_eval.sh [<label>]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"; PB="$REPO/frontier-evals/project/paperbench"
LABEL="${1:-$(date +%m%d_%H%M)}"
PB_JUDGE_PROVIDER="${PB_JUDGE_PROVIDER:-siliconflow}"
if [ "$PB_JUDGE_PROVIDER" = siliconflow ]; then
  set -a; . "${PB_JUDGE_ENV_FILE:-$HOME/Documents/env/siliconflow.env}"; set +a
  export OPENAI_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}" OPENAI_API_KEY="$SILICONFLOW_API_KEY"
  PB_JUDGE_MODEL="${PB_JUDGE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
  PB_STRUCTURED_PARSER_MODEL="${PB_STRUCTURED_PARSER_MODEL:-deepseek-ai/DeepSeek-V4-Pro}"; PB_JUDGE_THINKING="${PB_JUDGE_THINKING:-off}"  # the 09-21 16:40 caliber
elif [ "$PB_JUDGE_PROVIDER" = deepseek ]; then
  set -a; . "${PB_JUDGE_ENV_FILE:-$HOME/Documents/env/deepseek.env}"; set +a
  export OPENAI_BASE_URL="https://api.deepseek.com/v1" OPENAI_API_KEY="$DEEPSEEK_API_KEY"
  PB_JUDGE_MODEL="${PB_JUDGE_MODEL:-deepseek-flash}"
else echo "PB_JUDGE_PROVIDER=$PB_JUDGE_PROVIDER not supported here"; exit 2; fi
export PB_STRUCTURED_PARSER_MODEL="${PB_STRUCTURED_PARSER_MODEL:-$PB_JUDGE_MODEL}" PB_STRUCTURED_JSON_MODE="${PB_STRUCTURED_JSON_MODE:-json_object}"
export PB_JUDGE_WHOLE_CODEBASE="${PB_JUDGE_WHOLE_CODEBASE:-1}" PB_JUDGE_CONCURRENCY="${PB_JUDGE_CONCURRENCY:-20}"
[ "${PB_JUDGE_THINKING:-on}" = off ] && export PB_COMPLETER_EXTRA_BODY='{"enable_thinking": false}'
OUT="$REPO/runs/judge_eval/${LABEL}_$(echo "$PB_JUDGE_MODEL" | tr '/' '_')_tree${PB_JUDGE_WHOLE_CODEBASE}_think${PB_JUDGE_THINKING:-on}"
mkdir -p "$OUT"
echo "==== JudgeEval rice/0 · 裁判 $PB_JUDGE_MODEL @ $PB_JUDGE_PROVIDER · 解析器 $PB_STRUCTURED_PARSER_MODEL · 文件=$([ "$PB_JUDGE_WHOLE_CODEBASE" = 1 ] && echo 整棵树 || echo 每叶选10) · 思考=${PB_JUDGE_THINKING:-on} · $(date +%F\ %T) → $OUT"
cd "$PB"; export PATH="$HOME/Documents/search/.tools/bootstrap/bin:$HOME/.local/bin:$PATH"
uv run python -m paperbench.scripts.run_judge_eval judge=simple example_ids=rice/0 code_only=True output_dir="$OUT" \
  completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
  completer_config.model="$PB_JUDGE_MODEL" 2>&1 | grep -v "not found in tiktoken" | tail -15
