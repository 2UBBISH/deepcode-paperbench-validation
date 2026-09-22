#!/usr/bin/env bash
# ============================================================
# PaperBench 统一判分器(论文无关)
# 判完 ~/pb_submissions/<PAPER>/ 下的全部提交。
#
# ⚠️ 关键机制(官方文档没写、实测于 solver.py:140-147):
#    每个 task 实例只 `pop()` 一份提交。要判 N 份就必须 `paperbench.n_tries=N`,
#    否则只有 1 份被判、其余**无声忽略**。本脚本自动数目录并设置 n_tries。
#
# ⚠️ 两个必踩的坑:
#    ① ~/pb_submissions/ 下**每个子目录名都必须是合法 paper id**,
#       放个 fre_archive 之类的会在配置校验阶段直接失败。归档请放到 ~/pb_submissions_archive/。
#    ② 判分需要 Docker 在跑(LocalConfig 起沙箱),否则 sanity check 失败。
#
# 用法: PAPER=rice bash run_grade.sh        # 判分(花钱)
#       PAPER=rice DRY=1 bash run_grade.sh  # 只做检查与报价
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PAPER="${PAPER:-fre}"
PB="$REPO/frontier-evals/project/paperbench"
SUB_ROOT="$HOME/pb_submissions/$PAPER"
OUT="$REPO/runs/$PAPER"
DRY="${DRY:-0}"

echo "==== [0/4] 前置检查 ===="
docker info >/dev/null 2>&1 \
  || { echo "  ❌ Docker 未运行 —— 判分要用它起沙箱。先启动 Docker Desktop"; exit 1; }
echo "  ✅ Docker 在跑"
for d in "$HOME"/pb_submissions/*/; do
  n=$(basename "$d")
  [ -d "$PB/data/papers/$n" ] \
    || { echo "  ❌ ~/pb_submissions/$n 不是合法 paper id,判分器会拒绝。请移到 ~/pb_submissions_archive/"; exit 1; }
done
echo "  ✅ 提交根目录只含合法 paper id"

echo "==== [1/4] 提交清点与非空校验 ===="
[ -d "$SUB_ROOT" ] || { echo "❌ 没有 $SUB_ROOT"; exit 1; }
N=0; BAD=0
for d in "$SUB_ROOT"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  files=$(find "$d" -type f -not -path "*/.git/*" | wc -l)
  code=$(find "$d" -type f -name "*.py" -not -path "*/.git/*" | wc -l)
  printf "  %-14s 文件 %-4s python %-4s" "$name" "$files" "$code"
  if [ "$files" -lt 3 ]; then echo "  ❌ 过薄,判分是浪费钱"; BAD=1; else echo "  ✅"; fi
  N=$((N+1))
done
[ "$N" -gt 0 ] || { echo "❌ 没有任何提交目录"; exit 1; }
[ "$BAD" -eq 0 ] || { echo "❌ 存在过薄提交,先处理再判分"; exit 1; }

echo ""
echo "==== [2/4] 预估与闸门 ===="
echo "  论文     : $PAPER"
echo "  提交份数 : $N  → paperbench.n_tries=$N(不设则只判 1 份!)"
echo "  成本预估 : Paratera 实测约 ¥38/份；DeepSeek 官方（前缀缓存）估 ¥5–8/份 → 本次 $N 份"
echo "  闸门     : 本计划总预算 ¥600(2026-08-28 由 ¥500 上调)"
echo "  ⚠️ 已判过的提交请先移出 $SUB_ROOT,否则会重判、白花钱"
if [ "$DRY" = "1" ]; then
  echo ""; echo "DRY=1 → 到此为止,未花钱。去掉 DRY 即真判分。"; exit 0
fi

echo ""
# 裁判模型：PB_JUDGE_MODEL（默认 V4-Pro）。Flash 在 JudgeEval rice/0 上与 Pro 同准（0.719）但偏宽 2.2 pp（Pro 偏严 9 pp），
# 两种裁判的分数不能混表；二级解析器 PB_STRUCTURED_PARSER_MODEL 必须留 V4-Pro（Flash 对 response_format 返回坏 JSON）。
# 2026-09-21 起裁判走 DeepSeek 官方（PB_JUDGE_PROVIDER=deepseek，默认）：同一个 Flash 模型，官方 serving 自动前缀缓存——每叶提示词
# 95% 是同一前缀（系统 + 论文 + addendum + 代码），命中部分按缓存价计，一份从约 ¥38 降到约 ¥5–8。key 只从 PB_JUDGE_ENV_FILE
# （默认 ~/Documents/env/deepseek.env，DEEPSEEK_API_KEY）经环境变量注入，不写进 .env，不打印；load_dotenv 不覆盖已有环境变量。
# PB_JUDGE_PROVIDER=paratera 走旧路（.env 里的 OPENAI_BASE_URL / OPENAI_API_KEY，解析器 DeepSeek-V4-Pro）。
PB_JUDGE_PROVIDER="${PB_JUDGE_PROVIDER:-siliconflow}"  # owner 09-21 13:00: SiliconFlow (new key in ~/Documents/env/siliconflow.env; prefix cache verified)
if [ "$PB_JUDGE_PROVIDER" = siliconflow ]; then
  # owner 09-21: SiliconFlow (api.siliconflow.cn) — judge DeepSeek-V4-Flash (whole tree, thinking off), parser DeepSeek-V4-Pro;
  # key from ~/Documents/env/siliconflow.env. Caliber fixed 09-21 16:40 after JudgeEval rice/0: Flash judge thinking off = 0.70–0.72
  # accuracy (same as the top-10 / Paratera caliber's 0.719), and only the Pro parser (json_object, instance-not-schema guide)
  # keeps every leaf valid — the Flash parser voids ~5% of the "Score: 0" texts at random (9/178, 23/178 with thinking off).
  PB_JUDGE_ENV_FILE="${PB_JUDGE_ENV_FILE:-$HOME/Documents/env/siliconflow.env}"
  [ -f "$PB_JUDGE_ENV_FILE" ] || { echo "  ❌ 裁判 key 文件不存在: $PB_JUDGE_ENV_FILE"; exit 1; }
  set -a; . "$PB_JUDGE_ENV_FILE"; set +a
  [ -n "${SILICONFLOW_API_KEY:-}" ] || { echo "  ❌ $PB_JUDGE_ENV_FILE 里没有 SILICONFLOW_API_KEY"; exit 1; }
  export OPENAI_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1}" OPENAI_API_KEY="$SILICONFLOW_API_KEY"
  PB_JUDGE_MODEL="${PB_JUDGE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
  export PB_STRUCTURED_PARSER_MODEL="${PB_STRUCTURED_PARSER_MODEL:-deepseek-ai/DeepSeek-V4-Pro}" PB_STRUCTURED_JSON_MODE="${PB_STRUCTURED_JSON_MODE:-json_object}"
  PARSER="$PB_STRUCTURED_PARSER_MODEL"
  PB_JUDGE_THINKING="${PB_JUDGE_THINKING:-off}"
elif [ "$PB_JUDGE_PROVIDER" = deepseek ]; then
  PB_JUDGE_ENV_FILE="${PB_JUDGE_ENV_FILE:-$HOME/Documents/env/deepseek.env}"
  [ -f "$PB_JUDGE_ENV_FILE" ] || { echo "  ❌ 裁判 key 文件不存在: $PB_JUDGE_ENV_FILE"; exit 1; }
  set -a; . "$PB_JUDGE_ENV_FILE"; set +a
  [ -n "${DEEPSEEK_API_KEY:-}" ] || { echo "  ❌ $PB_JUDGE_ENV_FILE 里没有 DEEPSEEK_API_KEY"; exit 1; }
  export OPENAI_BASE_URL="https://api.deepseek.com/v1" OPENAI_API_KEY="$DEEPSEEK_API_KEY" PB_STRUCTURED_PARSER_MODEL="deepseek-flash"  # owner 09-21: everything deepseek-flash
  export PB_STRUCTURED_JSON_MODE=json_object  # the official API refuses json_schema (both models); json_object + schema-in-prompt instead
  PB_JUDGE_MODEL="${PB_JUDGE_MODEL:-deepseek-flash}"
  PARSER="$PB_STRUCTURED_PARSER_MODEL"
else
  PARSER=$(grep -E '^PB_STRUCTURED_PARSER_MODEL=' "$PB/.env" 2>/dev/null | cut -d= -f2)
  [ "$PARSER" = "DeepSeek-V4-Pro" ] || { echo "  ❌ $PB/.env 的 PB_STRUCTURED_PARSER_MODEL=$PARSER，必须是 DeepSeek-V4-Pro（README §6）"; exit 1; }
  PB_JUDGE_MODEL="${PB_JUDGE_MODEL:-DeepSeek-V4-Pro}"
fi
# PB_JUDGE_WHOLE_CODEBASE=1 (default with the official / SiliconFlow judges, 2026-09-21): every leaf sees the whole
# submission in one fixed order, no per-leaf file ranking → the 8–10万-token prefix is byte-identical across a
# submission's leaves and a prefix cache covers ~98% of it. 0 = upstream behaviour (rank, top 10 per leaf) — a different caliber.
[ "$PB_JUDGE_PROVIDER" = paratera ] || export PB_JUDGE_WHOLE_CODEBASE="${PB_JUDGE_WHOLE_CODEBASE:-1}"
# PB_JUDGE_THINKING=off sends enable_thinking:false with every judge / parser request (SiliconFlow honours it for
# V4-Flash: 0 reasoning tokens); the structured parser keeps the server default regardless (completer). SiliconFlow default
# off (09-21 16:40, JudgeEval: same accuracy, a third of the output cost); fre/line3's 0.921 of 12:24 was judged thinking on.
[ "${PB_JUDGE_THINKING:-on}" = off ] && export PB_COMPLETER_EXTRA_BODY='{"enable_thinking": false}'

echo "==== [3/4] 判分(code_only · 裁判 $PB_JUDGE_MODEL @ $PB_JUDGE_PROVIDER 恒定 · 解析器 $PARSER · 文件=$([ "${PB_JUDGE_WHOLE_CODEBASE:-0}" = 1 ] && echo 整棵树 || echo 每叶选10) · 思考=${PB_JUDGE_THINKING:-on})$(date +%F\ %T) ===="
cd "$PB"
export PATH="$HOME/.local/bin:$PATH"
# macOS Docker Desktop serves ~/.docker/run/docker.sock, not /var/run/docker.sock: tell docker-py and the sandbox.
export DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}"
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=$PAPER \
    paperbench.n_tries=$N \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.solver.computer_runtime.env.docker_host=$DOCKER_HOST \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model="$PB_JUDGE_MODEL" \
    paperbench.judge.code_only=True \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder 2>&1 | tail -20

echo ""
echo "==== [4/4] 结果与有效性核验 $(date +%F\ %T) ===="
# [4/4] reads THIS paper's newest run group, not the newest group of any paper: with several papers graded at once
# (night of 09-21/22, four workers) `ls -t runs/ | head -1` was another paper's still-empty group and nothing was
# copied ("NO grade.json" for a grading that had finished). A grading's own group is the newest one holding a
# <PAPER>_<run-id> directory.
G=$(ls -td runs/*/${PAPER}_* 2>/dev/null | head -1 | xargs -I{} dirname {} | xargs -I{} basename {})
[ -n "$G" ] || G=$(ls -t runs/ | head -1)
mkdir -p "$OUT/grades"
python3 - "$PB/runs/$G" "$OUT/grades" <<'PY'
import json, glob, os, shutil, sys
grp, out = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(grp, '*', 'grade.json'))):
    jo = json.load(open(f))['paperbench_result']['judge_output']
    bad, n, s = jo['num_invalid_leaf_nodes'], jo['num_leaf_nodes'], jo['score']
    tag = 'OK' if bad <= 2 else f'❌作废(无效叶 {bad}/{n} —— 判分中途出错,分数被压低,不可用)'
    print(f"  {os.path.basename(os.path.dirname(f))[:20]}  score={s:.4f}  叶={n}  无效叶={bad}  {tag}")
    shutil.copy(f, os.path.join(out, f"{os.path.basename(os.path.dirname(f))}.grade.json"))
print(f"\n  判分结果已复制到 {out}")
PY
echo "==== 判分 JSON 在 $OUT/grades（不入库）；数字请登记到 docs/RESULTS-HISTORY.md ===="
