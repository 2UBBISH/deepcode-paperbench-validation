#!/usr/bin/env bash
# 下载阶段观察哨:每 60s 查一次,仅在「仓库数变化 / StateMask 状态变化 / 下载阶段收尾 / 进程死亡」时输出。
# 下载阶段一结束就给出裁决并退出(退出码 9 让外层停表)。
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; export DEEPEVOL_ROOT="$R"
prev=""
while true; do
  T=$(ls -d "$R"/DeepCode/deepcode_lab/tasks/paper_*/ 2>/dev/null | head -1)
  DRV=$(pgrep -f "stage_b_drive[r]\.py" | head -1)
  L=$(ls -1t "$R"/deepcode_test/rice/logs/rice_trial1_deepcode_*.log 2>/dev/null | head -1)

  if [ -z "$DRV" ]; then
    echo "❌ driver 已退出 —— trial1 中止。日志尾部:"; tail -3 "$L" 2>/dev/null | cut -c1-160; exit 9
  fi
  [ -n "$T" ] || { sleep 60; continue; }

  NREPO=$(find "$T/code_base" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  # 逐仓库:检出文件数(不含 .git)
  detail=""
  for d in "$T"code_base/*/; do
    n=$(basename "$d"); f=$(find "$d" -type f -not -path '*/.git/*' 2>/dev/null | wc -l)
    detail="$detail $n:$f"
  done
  DONE_FILE=$([ -f "$T/github_download.txt" ] && echo yes || echo no)
  IDX_START=$(grep -ac "Starting code indexing process" "$L" 2>/dev/null)

  # 下载阶段是否收尾。判据不再盯某个具体仓库(候选清单每轮由模型自由提名,
  # 缺哪个属于要测量的轮间方差);只有「网络类失败」才是外部污染,值得重试。
  if [ "$DONE_FILE" = yes ] || [ "${IDX_START:-0}" -gt 0 ]; then
    echo "===== 下载阶段收尾 ====="
    echo "  实际落地 $NREPO 个:$detail"
    [ -f "$T/github_download.txt" ] && sed 's/^/    /' "$T/github_download.txt"
    NETFAIL=$(grep -aciE "TLS|GnuTLS|timed out|timeout|connection reset|recv error" "$T/github_download.txt" 2>/dev/null)
    if [ "${NETFAIL:-0}" -gt 0 ]; then
      echo "  ⚠️ 汇总里出现网络类失败($NETFAIL 处)—— 属外部污染,建议重试本轮"
    else
      echo "  ✅ 无网络类失败 —— 下成几个属模型自主选择,是要测量的方差,继续跑"
    fi
    exit 9
  fi

  cur="$NREPO|$detail"
  if [ "$cur" != "$prev" ]; then
    echo "[下载中] 已落地 $NREPO 个 |$detail"
    prev="$cur"
  fi
  sleep 60
done
