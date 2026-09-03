#!/usr/bin/env bash
# 索引阶段哨兵:每 3 分钟查一次,只在「新一趟索引开始」时输出,并对超大趟次告警。
# 每个仓库一趟,趟首会打 "Analyzing file 1/N" —— N 决定这一趟要跑多久(实测 54 秒/文件)。
# google-research 有 8885 个 py 文件,若不筛选会撞穿 14h 硬顶,故设 500 为告警线。
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; export DEEPEVOL_ROOT="$R"
PAPER="${PAPER:-rice}"
LIMIT="${LIMIT:-500}"
seen=""
while true; do
  L=$(ls -1t "$R"/deepcode_test/$PAPER/logs/${PAPER}_trial*_deepcode_*.log 2>/dev/null | head -1)
  DRV=$(pgrep -f "stage_b_drive[r]\.py" | head -1)
  [ -n "$L" ] || { sleep 180; continue; }

  if [ -z "$DRV" ]; then
    echo "❌ driver 已退出。日志尾部:"; tail -3 "$L" | cut -c1-160; exit 9
  fi

  # 每趟的总量(去重后按出现顺序)
  cur=$(grep -aoE "Analyzing file 1/[0-9]+" "$L" | sed 's|Analyzing file 1/||' | awk '!x[$0]++' | paste -sd,)
  if [ "$cur" != "$seen" ]; then
    new=${cur##*,}
    now=$(grep -ao "Analyzing file [0-9]*/[0-9]*" "$L" | tail -1)
    T=$(ls -d "$R"/DeepCode/deepcode_lab/tasks/paper_*/ 2>/dev/null | head -1)
    done_repo=$(ls -1 "$T"indexes/*_index.json 2>/dev/null | wc -l)
    if [ -n "$seen" ] && [ "${new:-0}" -gt "$LIMIT" ]; then
      hrs=$(awk "BEGIN{printf \"%.1f\", $new*54/3600}")
      echo "🚨 新一趟索引 $new 个文件 —— 超过 $LIMIT 告警线,按 54 秒/文件约需 ${hrs} 小时,会撞穿 14h 硬顶。建议掐掉本轮。"
      echo "   当前: $now | 已完成仓库 $done_repo | 各趟量: $cur"
    else
      echo "[索引] 新一趟 ${new} 个文件 | 当前 $now | 已完成仓库 $done_repo | 各趟量: $cur"
    fi
    seen="$cur"
  fi
  sleep 180
done
