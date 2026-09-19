#!/usr/bin/env bash
# One-time setup for the 0919 desktop batch: check tools, check the 20 papers are real text, block every paper's
# blacklisted repository in git (so an agent cannot clone the authors' code), create results/ and work/. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "==== 工具 ===="
for t in git python3 pbcopy; do command -v $t >/dev/null && echo "  ✅ $t" || { echo "  ❌ 缺 $t"; [ $t = pbcopy ] || exit 1; }; done
echo "==== 数据集（20 篇，只有 paper.md / addendum.md / blacklist.txt）===="
n=0; for d in "$ROOT"/data/papers/*/; do p=$(basename "$d"); [ -f "$d/paper.md" ] || { echo "  ❌ $p 缺 paper.md"; exit 1; }
  head -c 20 "$d/paper.md" | grep -q git-lfs && { echo "  ❌ $p/paper.md 是 LFS 指针"; exit 1; }; n=$((n+1)); done
[ "$n" = 20 ] && echo "  ✅ 20 篇就绪" || { echo "  ❌ 只有 $n 篇"; exit 1; }
[ -f "$ROOT/instructions/code_only_instructions.txt" ] && echo "  ✅ 官方题面 instructions/code_only_instructions.txt" || { echo "  ❌ 缺官方题面"; exit 1; }
echo "==== git 反抄袭封锁（每篇 blacklist.txt 里的仓库，含大小写与 ssh 变体）===="
for d in "$ROOT"/data/papers/*/; do p=$(basename "$d"); i=0
  while read -r url; do
    case "$url" in ""|none|None) continue ;; esac
    repo="${url#https://github.com/}"; repo="${repo#http://github.com/}"; repo="${repo%/}"; repo="${repo%.git}"; i=$((i+1))
    lower="$(echo "$repo" | tr 'A-Z' 'a-z')"; upper="$(echo "$repo" | tr 'a-z' 'A-Z')"
    for v in "$repo" "$lower" "$upper"; do git config --global "url.https://blocked.invalid/$p-$i-$v.insteadOf" "https://github.com/$v"; done
    git config --global "url.https://blocked.invalid/$p-$i-ssh.insteadOf" "git@github.com:$repo"
    echo "  🚫 $p: $repo"
  done < <(grep -vE '^\s*(#|$)' "$d/blacklist.txt")
  [ "$i" = 0 ] && echo "  ℹ️ $p: blacklist 为 none"
done
echo "  （只挡 git 克隆；网页上看到也不能抄——跑完 desktop_finish.sh 会 grep 一遍）"
mkdir -p "$ROOT/results" "$ROOT/work"; echo "==== ✅ results/（交回）与 work/（工作目录）已建 ===="
