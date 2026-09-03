#!/usr/bin/env bash
# 修复轮收尾链:等 trial_fx1/fx2 结束 → Paratera 裁判判 fre 批(fx1/fx2 + 旧4份)
# → 校验无效叶(金丝雀)→ 通过才判 rice 批(旧5份)。全程写台账。
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DEEPEVOL_ROOT="$R"
LED=$R/deepcode_test/fre/logs/fre_ledger_fix.txt
log(){ echo "[$(date +'%m-%d %H:%M:%S')] $*" | tee -a "$LED"; }

# ── 1. 等两轮结束(以台账收尾行 + 无进程为准)──
while true; do
  grep -q "修复验证两轮结束" "$LED" 2>/dev/null && ! pgrep -f "stage_b_drive[r]\.py" >/dev/null && break
  sleep 120
done
log "🔗 收尾链启动:两轮已结束,开始判分刷新(裁判=Paratera DeepSeek-V4-Pro)"

# ── 2. 余额探针(2000 token 真实请求)──
PR=$(curl -s -m 90 https://llmapi.paratera.com/v1/chat/completions \
  -H "Authorization: Bearer $(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.deepcode/credentials.json')))['connections']['paratera'])")" -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Pro","messages":[{"role":"user","content":"Count 1 to 300, one per line."}],"max_tokens":2000}')
echo "$PR" | grep -q '"choices"' || { log "❌ Paratera 探针失败,判分中止: $(echo "$PR"|head -c 120)"; exit 1; }
log "✅ 探针通过"

docker info >/dev/null 2>&1 || { log "❌ Docker 未跑,判分中止"; exit 1; }

# ── 3. fre 批:fx1/fx2(若摆卷成功)+ 旧 4 份复判 ──
for d in trial1 trial5 bare_v4 anchor; do
  [ -d "$HOME/pb_submissions_archive/fre_graded/$d" ] && cp -r "$HOME/pb_submissions_archive/fre_graded/$d" "$HOME/pb_submissions/fre/" 2>/dev/null
done
log "fre 批清单: $(ls -m $HOME/pb_submissions/fre/)"
PAPER=fre bash $R/deepcode_test/scripts/run_grade.sh > $R/deepcode_test/fre/logs/refresh_grade_fre.log 2>&1
G=$(ls -1t $R/frontier-evals/project/paperbench/runs/ | head -1)
BAD=$(python3 - "$R/frontier-evals/project/paperbench/runs/$G" <<'PY'
import json,glob,sys
mx=-1
for f in glob.glob(sys.argv[1]+'/*/grade.json'):
    try:
        jo=json.load(open(f)).get('paperbench_result',{}).get('judge_output')
        if jo: mx=max(mx,jo['num_invalid_leaf_nodes'])
    except Exception: pass
print(mx if mx>=0 else 999)
PY
)
log "fre 批完成(判分组 $G),最大无效叶=$BAD"
if [ "${BAD:-99}" -gt 2 ]; then
  log "❌ fre 批无效叶超标($BAD)—— 金丝雀失败,rice 批不判,等人工处理"
  exit 2
fi

# ── 4. rice 批:旧 5 份复判 ──
for d in trial2 trial3 bare_v4 trial_k1 trial_k2; do
  [ -d "$HOME/pb_submissions_archive/rice_graded/$d" ] && cp -r "$HOME/pb_submissions_archive/rice_graded/$d" "$HOME/pb_submissions/rice/" 2>/dev/null
done
log "rice 批清单: $(ls -m $HOME/pb_submissions/rice/ 2>/dev/null)"
PAPER=rice bash $R/deepcode_test/scripts/run_grade.sh > $R/deepcode_test/rice/logs/refresh_grade_rice.log 2>&1
G2=$(ls -1t $R/frontier-evals/project/paperbench/runs/ | head -1)
log "rice 批完成(判分组 $G2)"
log "🏁 收尾链全部完成:fre 组=$G rice 组=$G2"
