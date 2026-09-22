#!/usr/bin/env bash
# 判分进度快照(论文无关,自动认最新判分组)。结束时以退出码 9 收摊。
# 归属靠 tarball 里的 py 文件数认(metadata.json 不记提交名)。
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; export DEEPEVOL_ROOT="$R"
PB="$R/frontier-evals/project/paperbench"
G="$PB/runs/$(ls -1t "$PB/runs" | head -1)"
ALIVE=$(pgrep -f "run_grad[e]\.sh" | head -1)
DONE=$(ls -1 "$G"/*/grade.json 2>/dev/null | wc -l)
WANT=$(ls -1d "$G"/*_*/ 2>/dev/null | wc -l)

name_of() {
  local t; t=$(ls "$1"/submissions/*/submission.tar.gz 2>/dev/null | head -1)
  [ -f "$t" ] || { basename "$1" | cut -c1-14; return; }
  case "$(tar tzf "$t" 2>/dev/null | grep -c '\.py$')" in
    11) echo bare_v4 ;; 22) echo trial2 ;; 39) echo trial3 ;;
    36) echo trial_k1 ;; 32) echo trial_k2 ;; 18) echo bare_kimi ;;
    15) echo "fre:bare_v4/anchor" ;; 28) echo "fre:trial5" ;;
    *) basename "$1" | cut -c1-14 ;;
  esac
}

if [ -n "$ALIVE" ]; then
  ET=$(ps -o etime= -p "$ALIVE" 2>/dev/null | tr -d ' ')
  echo "[判分 $ET] 已出分 $DONE/$WANT"
else
  echo "[判分已结束] 出分 $DONE/$WANT"
  for f in "$G"/*/grade.json; do
    [ -f "$f" ] || continue
    d=$(dirname "$f")
    python3 -c "
import json
jo=json.load(open('$f'))['paperbench_result']['judge_output']
bad=jo['num_invalid_leaf_nodes']; n=jo['num_leaf_nodes']
flag='' if bad<=2 else '  ⚠️ 无效叶过多,分数不可用'
print('  $(name_of "$d"): score=%.4f | 叶=%d | 无效叶=%d%s' % (jo['score'], n, bad, flag))"
  done
  [ "$DONE" -lt "$WANT" ] && tail -3 $R/deepcode_test/rice/logs/rice_grade_console.log
  exit 9
fi
