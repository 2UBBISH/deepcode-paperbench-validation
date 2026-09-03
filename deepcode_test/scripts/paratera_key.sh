#!/usr/bin/env bash
# ============================================================
# Paratera API key 统一管理
#
# 为什么需要它:key 同时被两个消费方读取,手工改容易漏一处 ——
#   ① 底座  ~/.deepcode/credentials.json         connections.paratera
#   ② 裁判  frontier-evals/.../paperbench/.env    OPENAI_API_KEY
# 漏改的表现是「复现跑得好好的,判分全部 401」,而且要等判分启动才发现。
#
# 用法:
#   bash paratera_key.sh check              探活当前 key(2000 token 真实请求,约 ¥0.01)
#   bash paratera_key.sh set <新key>        探活通过后,原子写入上述两处(自动备份)
#   bash paratera_key.sh add <key> [备注]   把 key 加进备用池,不切换
#   bash paratera_key.sh list               列出备用池(打码显示)
#   bash paratera_key.sh next               当前 key 探活失败时,自动切到池里第一个可用的
#
# 备用池: ~/.deepcode/paratera_keys.json(chmod 600,不进仓库)
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CRED="$HOME/.deepcode/credentials.json"
POOL="$HOME/.deepcode/paratera_keys.json"
PB_ENV="$REPO/frontier-evals/project/paperbench/.env"
BASE_URL="https://llmapi.paratera.com/v1"
MODEL="${PARATERA_PROBE_MODEL:-DeepSeek-V4-Pro}"

mask() { local k="$1"; echo "${k:0:6}…${k: -4}"; }

current_key() {
  python3 -c "
import json,os,sys
p=os.path.expanduser('$CRED')
try: print(json.load(open(p))['connections'].get('paratera',''))
except Exception: print('')"
}

# 探活:发一个真实但极小的请求。返回 0=可用;非 0 并打印原因。
probe() {
  local key="$1" out code
  out=$(curl -s -m 90 -w '\n%{http_code}' "$BASE_URL/chat/completions" \
        -H "Authorization: Bearer $key" -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":2000}" 2>&1) || true
  code=$(printf '%s' "$out" | tail -1)
  case "$code" in
    200) echo "    ✅ HTTP 200,可用"; return 0 ;;
    402) echo "    ❌ HTTP 402 —— 余额不足"; return 1 ;;
    403)
      # Paratera 在余额耗尽后不返回 402,而是把 key 降级到免费档:
      # 付费模型 403 team_model_access_denied,免费模型仍 200,/v1/models 只剩 Flash 档。
      # 2026-09-03 实证。平台无余额查询端点(credit_grants / balance 等均 404)。
      if printf '%s' "$out" | grep -q "team_model_access_denied"; then
        local n_free
        n_free=$(curl -s -m 20 "$BASE_URL/models" -H "Authorization: Bearer $key" \
                 | python3 -c "import json,sys;print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "?")
        echo "    ❌ HTTP 403 team_model_access_denied —— 该 key 访问不到 $MODEL"
        echo "       可访问模型仅剩 $n_free 个。**最可能是余额耗尽**(Paratera 余额见底后不报 402,"
        echo "       而是把 key 降级到免费 Flash 档);其次才是团队权限变更。请充值或换 key。"
      else
        echo "    ❌ HTTP 403 —— $(printf '%s' "$out" | head -c 160)"
      fi
      return 1 ;;
    401) echo "    ❌ HTTP 401 —— key 无效"; return 1 ;;
    429) echo "    ⚠️ HTTP 429 —— 限流(key 本身可能有效,稍后重试)"; return 2 ;;
    *)   echo "    ❌ HTTP ${code:-?} —— $(printf '%s' "$out" | head -c 160)"; return 1 ;;
  esac
}

# 把 key 写进两个消费方(先备份,失败不留半吊子状态)
apply_key() {
  local key="$1" ts; ts=$(date +%m%d_%H%M%S)
  [ -f "$CRED" ] && cp "$CRED" "$CRED.bak_$ts"
  python3 - "$CRED" "$key" <<'PY'
import json,os,sys
p,k=os.path.expanduser(sys.argv[1]),sys.argv[2]
d=json.load(open(p)) if os.path.exists(p) else {"version":1,"connections":{}}
d.setdefault("connections",{})["paratera"]=k
tmp=p+".tmp"; json.dump(d,open(tmp,"w"),indent=2,ensure_ascii=False); os.replace(tmp,p); os.chmod(p,0o600)
PY
  echo "    ✅ 底座 ~/.deepcode/credentials.json"

  if [ -f "$PB_ENV" ]; then
    cp "$PB_ENV" "$PB_ENV.bak_$ts"
    python3 - "$PB_ENV" "$key" <<'PY'
import sys,os,re
p,k=sys.argv[1],sys.argv[2]
lines=open(p,encoding='utf-8').read().splitlines()
seen=False; out=[]
for l in lines:
    if re.match(r'^\s*OPENAI_API_KEY\s*=', l): out.append(f"OPENAI_API_KEY={k}"); seen=True
    else: out.append(l)
if not seen: out.append(f"OPENAI_API_KEY={k}")
tmp=p+".tmp"; open(tmp,'w',encoding='utf-8').write("\n".join(out)+"\n"); os.replace(tmp,p); os.chmod(p,0o600)
PY
    echo "    ✅ 裁判 paperbench/.env"
  else
    echo "    ⚠️ 未找到 $PB_ENV —— 只更新了底座;跑 setup.sh 后需重跑本命令"
  fi
}

pool_read() {
  [ -f "$POOL" ] || { echo "[]"; return; }
  python3 -c "
import json,os
try: print(json.dumps(json.load(open(os.path.expanduser('$POOL'))).get('keys',[])))
except Exception: print('[]')"
}

case "${1:-}" in
  check)
    k=$(current_key)
    [ -n "$k" ] || { echo "❌ credentials.json 里没有 paratera key"; exit 1; }
    echo "当前 key: $(mask "$k")"; probe "$k"; exit $?
    ;;

  set)
    NEW="${2:-}"; [ -n "$NEW" ] || { echo "用法: $0 set <新key>"; exit 1; }
    echo "探活新 key $(mask "$NEW") …"
    if probe "$NEW"; then
      echo "写入两处 …"; apply_key "$NEW"
      echo "完成。备份后缀 .bak_$(date +%m%d)*"
    else
      echo "❌ 新 key 探活未通过,未做任何修改"; exit 1
    fi
    ;;

  add)
    NEW="${2:-}"; NOTE="${3:-}"; [ -n "$NEW" ] || { echo "用法: $0 add <key> [备注]"; exit 1; }
    python3 - "$POOL" "$NEW" "$NOTE" <<'PY'
import json,os,sys
p,k,note=os.path.expanduser(sys.argv[1]),sys.argv[2],sys.argv[3]
d=json.load(open(p)) if os.path.exists(p) else {"keys":[]}
if any(e["key"]==k for e in d["keys"]): print("    ⏭ 已在池中"); sys.exit(0)
d["keys"].append({"key":k,"note":note})
tmp=p+".tmp"; json.dump(d,open(tmp,"w"),indent=2,ensure_ascii=False); os.replace(tmp,p); os.chmod(p,0o600)
print(f"    ✅ 已加入备用池(现有 {len(d['keys'])} 个)")
PY
    ;;

  list)
    echo "当前使用: $(mask "$(current_key)")"
    echo "备用池 ($POOL):"
    python3 - <<PY
import json,os
p=os.path.expanduser("$POOL")
if not os.path.exists(p): print("  (空)"); raise SystemExit
for i,e in enumerate(json.load(open(p)).get("keys",[]),1):
    k=e["key"]; print(f"  {i}. {k[:6]}…{k[-4:]}  {e.get('note','')}")
PY
    ;;

  next)
    cur=$(current_key)
    echo "当前 key $(mask "$cur") 探活:"
    if probe "$cur"; then echo "仍可用,无需切换"; exit 0; fi
    echo "尝试备用池 …"
    mapfile -t KEYS < <(pool_read | python3 -c "import json,sys;[print(e['key']) for e in json.load(sys.stdin)]")
    [ "${#KEYS[@]}" -gt 0 ] || { echo "❌ 备用池为空,先用 '$0 add <key>' 加入"; exit 1; }
    for k in "${KEYS[@]}"; do
      [ "$k" = "$cur" ] && continue
      echo "  试 $(mask "$k"):"
      if probe "$k"; then echo "  切换到该 key:"; apply_key "$k"; exit 0; fi
    done
    echo "❌ 备用池里没有可用 key"; exit 1
    ;;

  *)
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
