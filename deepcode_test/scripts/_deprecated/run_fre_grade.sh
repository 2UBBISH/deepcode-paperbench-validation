#!/usr/bin/env bash
# ============================================================
# fre 三方验证 · 统一判分器(F 系列 · 步骤 E)
# 一条命令判完 ~/pb_submissions/fre/ 下的全部提交(锚点 + 各 trial)。
#
# ⚠️ 关键机制(审计漏抓、实测于 solver.py:140-147):
#    每个 task 实例只 `pop()` 一份提交。要判 N 份就必须 `paperbench.n_tries=N`,
#    否则只有 1 份被判、其余**无声忽略**。本脚本自动数目录并设置 n_tries。
#
# 用法: bash run_fre_grade.sh        # 判分(会花钱,先看它打印的预估)
#       DRY=1 bash run_fre_grade.sh  # 只做检查与报价,不花钱
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER=fre
PB="$ROOT/frontier-evals/project/paperbench"
SUB_ROOT="$HOME/pb_submissions/$PAPER"
DRY="${DRY:-0}"

echo "==== [1/3] 提交清点与非空校验 ===="
[ -d "$SUB_ROOT" ] || { echo "❌ 没有 $SUB_ROOT,先跑 run_fre_trial.sh / 锚点"; exit 1; }

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
[ "$BAD" -eq 0 ] || { echo "❌ 存在过薄提交,先处理再判分(或手工移走)"; exit 1; }

echo ""
echo "==== [2/3] 预估与闸门 ===="
echo "  提交份数 : $N  → paperbench.n_tries=$N(不设则只判 1 份!)"
echo "  判分叶数 : 306/份(fre rubric 437 叶中 Code Development 占 70%)"
echo "  成本预估 : ¥45~60/份 → 本次约 ¥$((N*45))~$((N*60))"
echo "  闸门     : 本计划总预算 ¥600(2026-08-28 由 ¥500 上调);单轮 trial 熔断 ¥100"
if [ -f "$ROOT/FRE_VALIDATION_PLAN.md" ]; then
  echo "  提示     : 判分前请确认所有 trial 已完成,漏判的提交不会被自动补判"
fi
if [ "$DRY" = "1" ]; then
  echo ""
  echo "DRY=1 → 到此为止,未花钱。去掉 DRY 即真判分。"
  exit 0
fi

echo ""
echo "==== [3/3] 判分(code_only · DeepSeek-V4-Pro 裁判恒定)$(date +%F\ %T) ===="
cd "$PB"
export PATH="$HOME/.local/bin:$PATH"
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=fre \
    paperbench.n_tries=$N \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='deepseek-ai/DeepSeek-V4-Pro' \
    paperbench.judge.code_only=True \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder 2>&1 | tail -20

G=$(ls -t runs/ | head -1)
echo ""
echo "==== 完成 $(date +%F\ %T)。结果: $PB/runs/$G/fre_*/grade.json ===="
for f in "$PB/runs/$G"/fre_*/grade.json; do
  [ -f "$f" ] || continue
  python3 -c "
import json,sys
d=json.load(open('$f'))
print('  ', '$f'.split('/')[-2], '→ score:', d.get('score'), '| leaves:', d.get('num_leaf_nodes'), '| failed:', d.get('n_gradings_failed'))" 2>/dev/null || true
done
echo "==== 记得回填 FRE_VALIDATION_PLAN.md §6 ===="
