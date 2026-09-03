#!/usr/bin/env bash
# ============================================================
# 评分知识泄漏扫描 —— 跑自优化循环前的强制闸门
#
# 为什么存在:修复轮 trial_fx1/fx2 的提示词里只是出现了一句
#   "Graders assign separate credit to each baseline; omitting them forfeits those points"
# 两轮产物就整体作废(docs/REVIEW_local_changes_2026-09-03.md)。
# 评分表知识一旦进入流水线,产出的分数就不再是对复现能力的测量。
#
# 用法:
#   bash check_no_rubric_leak.sh                 # 扫默认范围(DeepCode 提示词 + 脚本)
#   bash check_no_rubric_leak.sh <目录> [...]     # 扫指定目录
#   STRICT=1 bash check_no_rubric_leak.sh        # 连 rubric 文件存在本身也算失败
#
# 退出码: 0 = 干净;1 = 发现泄漏
# ============================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

# 默认扫描范围:会进入 LLM 提示词或流水线逻辑的地方
if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # 只扫「会成为复现 agent 的 LLM 输入」的地方。
  # 判分脚本(run_grade.sh、monitor/、refresh_*)属于度量侧,提到 judge 是本分,不在扫描范围。
  TARGETS=(
    "$REPO/DeepCode/workflows"
    "$REPO/DeepCode/prompts"
    "$REPO/DeepCode/tools"
    "$REPO/deepcode_test/docs/CC_FRE_PROMPT.txt"
    "$REPO/deepcode_test/rice/workspaces/cc_dsv4_run/PROMPT.txt"
    "$REPO/deepcode_test/rice/workspaces/cc_kimi_run/PROMPT.txt"
  )
fi

# 评分体系专有词。刻意不含 "score"/"评分" —— 那些词在正常工程语境里太常见,
# 会淹没真正的信号。这里只留 PaperBench 评分体系的标志性词汇。
# 只留「评分体系知识」的标志性表述。judge/leaf_node 等词在度量侧脚本里合法,
# 故不列入 —— 扫描范围已排除度量侧,这里再收窄可避免误报淹没真信号。
PATTERNS='grader|Graders|rubric|forfeit|separate credit|weighted score|judging criteri|评分表|判分树|叶子权重|评分细则'

echo "==== 评分知识泄漏扫描 ===="
echo "  仓库根: $REPO"
FOUND=0
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] || { echo "  ⏭ 跳过(不存在): ${t#$REPO/}"; continue; }
  # 排除本脚本自身与说明文档(它们理应提到这些词)
  hits=$(grep -rInE "$PATTERNS" "$t" \
          --include='*.py' --include='*.sh' --include='*.txt' --include='*.yaml' --include='*.yml' --include='*.json' \
          --exclude='check_no_rubric_leak.sh' --exclude='OPTIMIZER_NOTICE.md' \
          --exclude-dir='__pycache__' --exclude-dir='.git' 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "  ❌ ${t#$REPO/}"
    echo "$hits" | head -20 | sed 's/^/       /' | cut -c1-160
    n=$(echo "$hits" | wc -l)
    [ "$n" -gt 20 ] && echo "       … 另有 $((n-20)) 处"
    FOUND=1
  else
    echo "  ✅ ${t#$REPO/}"
  fi
done

# rubric 文件是否在工作树里(默认只提醒,STRICT=1 时判失败)
RUB="$REPO/paperbench_changes/rubrics"
if [ -d "$RUB" ]; then
  if [ "${STRICT:-0}" = "1" ]; then
    echo "  ❌ STRICT: rubric 文件存在于工作树 ($RUB)"
    echo "       优化器可读到判分树。移出方法:"
    echo "       git rm -r --cached paperbench_changes/rubrics && echo 'paperbench_changes/rubrics/' >> .git/info/exclude"
    FOUND=1
  else
    echo "  ⚠️  rubric 文件在工作树里($RUB)"
    echo "       它是给人事后核对用的。若要跑自优化循环,请先移出(见 docs/OPTIMIZER_NOTICE.md §4),"
    echo "       或用 STRICT=1 让本脚本把它判为失败。"
  fi
fi

echo
if [ "$FOUND" -eq 0 ]; then
  echo "✅ 未发现评分知识泄漏"
else
  echo "❌ 发现泄漏 —— 不要在此状态下跑优化循环"
  echo "   背景:docs/OPTIMIZER_NOTICE.md §4、docs/REVIEW_local_changes_2026-09-03.md"
fi
exit "$FOUND"
