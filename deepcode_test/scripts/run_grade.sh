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
OUT="$HERE/../$PAPER"
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
echo "  成本预估 : 实测约 ¥38/份 → 本次约 ¥$((N*38))(脚本报价上限 ¥$((N*60)))"
echo "  闸门     : 本计划总预算 ¥600(2026-08-28 由 ¥500 上调)"
echo "  ⚠️ 已判过的提交请先移出 $SUB_ROOT,否则会重判、白花钱"
if [ "$DRY" = "1" ]; then
  echo ""; echo "DRY=1 → 到此为止,未花钱。去掉 DRY 即真判分。"; exit 0
fi

echo ""
echo "==== [3/4] 判分(code_only · DeepSeek-V4-Pro 裁判恒定)$(date +%F\ %T) ===="
cd "$PB"
export PATH="$HOME/.local/bin:$PATH"
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=$PAPER \
    paperbench.n_tries=$N \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model="${PB_JUDGE_MODEL:-DeepSeek-V4-Pro}" \
    paperbench.judge.code_only=True \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder 2>&1 | tail -20

echo ""
echo "==== [4/4] 结果与有效性核验 $(date +%F\ %T) ===="
G=$(ls -t runs/ | head -1)
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
echo "==== 记得回填 docs/HANDOFF_FRE.md ===="
