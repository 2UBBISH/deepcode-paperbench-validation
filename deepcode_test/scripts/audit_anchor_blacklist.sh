#!/usr/bin/env bash
# ============================================================
# 锚点作弊审计(F 系列 · 步骤 B 收尾)
# git insteadOf 只挡 git clone,挡不住网页抓取/curl。本脚本事后扫描锚点
# agent 的完整 transcript,确认它没有读过 fre 的官方实现。
#
# 用法: bash audit_anchor_blacklist.sh <workflow_transcript_dir>
#   例: bash audit_anchor_blacklist.sh ~/.claude/projects/*/subagents/workflows/wf_1b24331a-705
# ============================================================
set -uo pipefail
DIR="${1:-}"
[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "用法: bash $0 <transcript_dir>"; exit 1; }

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DEEPEVOL_ROOT="$R"
SUB="$R/anchor_fre/submission"
echo "==== 锚点作弊审计 $(date +%F\ %T) ===="
echo "transcript: $DIR"
echo ""

# 1) 黑名单 URL 的各种形态
echo "--- [1] transcript 中是否出现黑名单仓库 ---"
PAT='kvfrans/fre|kvfrans%2Ffre|github\.com/kvfrans'
HITS=$(grep -rEoi "$PAT" "$DIR" 2>/dev/null | wc -l)
if [ "$HITS" -eq 0 ]; then
  echo "  ✅ 0 命中"
else
  echo "  ⚠️ $HITS 处命中,逐条上下文如下(需人工判定是"读了"还是"被挡后放弃"):"
  grep -rEoi ".{100}($PAT).{140}" "$DIR" 2>/dev/null | head -12
fi

# 2) 是否尝试过 clone/抓取(即便被挡,也要记录在案)
echo ""
echo "--- [2] 是否出现抓取动作(WebFetch / curl / wget / git clone) ---"
for k in WebFetch WebSearch "git clone" curl wget; do
  n=$(grep -rFoi "$k" "$DIR" 2>/dev/null | wc -l)
  printf "  %-12s %s\n" "$k" "$n"
done

# 3) 产物里是否混入了上游代码指纹
echo ""
echo "--- [3] 提交物中是否出现上游仓库指纹 ---"
if [ -d "$SUB" ]; then
  F=$(grep -rEli "kvfrans|github\.com/kvfrans" "$SUB" --exclude-dir=.git 2>/dev/null | head -5)
  if [ -z "$F" ]; then echo "  ✅ 提交物无上游指纹"; else echo "  ⚠️ 以下文件提到上游:"; echo "$F" | sed 's/^/     /'; fi
else
  echo "  (提交目录不存在)"
fi

# 4) 提交物合规性(官方要求:git 仓库 + README.md)
echo ""
echo "--- [4] 提交物合规性 ---"
if [ -d "$SUB" ]; then
  echo "  文件数      : $(find "$SUB" -type f -not -path '*/.git/*' | wc -l)"
  echo "  python 文件 : $(find "$SUB" -type f -name '*.py' -not -path '*/.git/*' | wc -l)"
  echo "  git 仓库    : $([ -d "$SUB/.git" ] && echo yes || echo '❌ no(官方要求是 git 仓库)')"
  echo "  README.md   : $([ -f "$SUB/README.md" ] && echo yes || echo '❌ 缺失(官方明确要求)')"
  echo "  体积        : $(du -sh "$SUB" 2>/dev/null | cut -f1)"
  if [ -d "$SUB/.git" ]; then
    echo "  未跟踪文件  : $(cd "$SUB" && git status --porcelain 2>/dev/null | grep -c '^??')  ← 官方会 git clean -fd,未跟踪即丢失"
  fi
else
  echo "  ❌ 提交目录不存在"
fi
echo ""
echo "==== 审计结束 ===="
