#!/usr/bin/env bash
# One bare-arm run, start to finish: workspace + PROMPT.txt → proxy (thinking off, key from the env file) → the CLI
# non-interactively → continue rounds while nothing is committed → audit → copy into the judge's pool.
#
#   bash run_bare.sh <codex|claude> <paper-id> [--hours N] [--root DIR] [--pool DIR] [--max-continues N] [--no-pool]
#
# Caliber (docs/CODEDEV-ARMS.md §4): DeepSeek-V4-Flash, thinking off (the proxy injects thinking:{type:disabled} and the
# log proves it: reasoning_tokens / reasoning_items (Codex) or thinking_blocks (Claude Code) are 0 on every line),
# PaperBench's official Code-Dev instructions + its ADDITIONAL NOTES, no rubric, blacklist audited afterwards.
# The CLIs' own auth is bypassed: the proxy authenticates upstream from PARATERA_API_KEY in ~/Documents/env/paratera.env
# (sourced only in the proxy's subshell; never printed). Codex's config.toml is left alone — model / provider / base_url
# are overridden on the command line (-c) for this process only; Claude Code gets everything via env vars.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"; shift 2
HOURS=""; ROOT="$HOME/Documents/env/bare-0919"; POOL="$HOME/pb_submissions"; MAX_CONT=5; TO_POOL=1
while [ $# -gt 0 ]; do case "$1" in
  --hours) HOURS="$2"; shift 2 ;; --root) ROOT="$2"; shift 2 ;; --pool) POOL="$2"; shift 2 ;;
  --max-continues) MAX_CONT="$2"; shift 2 ;; --no-pool) TO_POOL=0; shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
[ "$ARM" = codex ] || [ "$ARM" = claude ] || { echo "arm must be codex or claude"; exit 2; }
ENV_FILE="$HOME/Documents/env/paratera.env"; [ -f "$ENV_FILE" ] || { echo "❌ $ENV_FILE missing"; exit 1; }
export PATH="$HOME/.local/node-v24.21.0/bin:$PATH"   # where npm -g put codex / claude on this Mac
command -v "$ARM" >/dev/null || { echo "❌ $ARM CLI not on PATH"; exit 1; }
MODEL="DeepSeek-V4-Flash"
PORT=$([ "$ARM" = codex ] && echo 8787 || echo 8788)
WS="$ROOT/$PAPER-$ARM"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WS/RUN_NOTES.md"; }

# ---- 1. workspace + prompt (refuses a non-empty directory: one workspace per run) ----
mkdir -p "$ROOT"
bash "$HERE/render_prompt.sh" "$PAPER" "$WS" ${HOURS:+--hours "$HOURS"} | tee "$WS/render.log"
grep -q PROMPT_OK "$WS/render.log" || { echo "❌ prompt not rendered"; exit 1; }
{ echo "# $PAPER / $ARM — $(date '+%F %T')"; echo "- cli: $($ARM --version 2>&1 | head -1)"; echo "- model: $MODEL, thinking off via proxy :$PORT"; echo "- validation repo: $(git -C "$HERE" rev-parse --short HEAD)"; echo "- time limit in prompt: ${HOURS:-none (no_time_limit_template)}"; } > "$WS/RUN_NOTES.md"

# ---- 2. harness hygiene: nothing but the benchmark's input may reach the agent ----
if [ "$ARM" = codex ]; then
  [ ! -s "$HOME/.codex/AGENTS.md" ] || { echo "❌ ~/.codex/AGENTS.md is not empty"; exit 1; }
  log "- codex: model_reasoning_effort=$(grep -m1 '^model_reasoning_effort' "$HOME/.codex/config.toml" | cut -d= -f2- | tr -d ' "' || echo default) (Codex's own setting, sent as reasoning.effort and logged)"
else
  [ ! -f "$HOME/.claude/CLAUDE.md" ] || { echo "❌ ~/.claude/CLAUDE.md exists; move it aside for the run"; exit 1; }
fi
[ ! -f "$WS/AGENTS.md" ] && [ ! -f "$WS/CLAUDE.md" ] || { echo "❌ instruction files in the workspace"; exit 1; }

# ---- 3. proxy ----
( set -a; . "$ENV_FILE"; set +a
  PROXY_THINKING=disabled PROXY_UPSTREAM_KEY_ENV=PARATERA_API_KEY \
  exec python3 "$HERE/paratera_proxy.py" "$PORT" "$WS/proxy_requests.log" > "$WS/proxy_stdout.log" 2>&1 ) &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
sleep 1; kill -0 $PROXY_PID 2>/dev/null || { echo "❌ proxy did not start (port $PORT busy?)"; cat "$WS/proxy_stdout.log"; exit 1; }
log "- proxy pid $PROXY_PID on :$PORT"

# ---- 4. the run, then continue rounds while nothing is committed ----
PROMPT="$(cat "$WS/PROMPT.txt")"
CONTINUE="$(cat "$WS/CONTINUE.txt")"
SESSION="$(python3 -c 'import uuid; print(uuid.uuid4())')"
committed() { git -C "$WS/submission" rev-parse --verify HEAD >/dev/null 2>&1; }
run_codex() {  # $1 = round (0 = first), $2 = prompt
  local common=(-c "model=$MODEL" -c model_provider=custom -c "model_providers.custom.base_url=http://127.0.0.1:$PORT/v1"
                -c sandbox_workspace_write.network_access=true --skip-git-repo-check --approve-for-me --json)
  if [ "$1" = 0 ]; then
    codex exec -C "$WS" "${common[@]}" -o "$WS/last_message.$1.txt" "$2" < /dev/null   # stdin closed: codex exec otherwise waits on it
  else
    codex exec resume "${common[@]}" --last -o "$WS/last_message.$1.txt" "$2" < /dev/null
  fi
}
run_claude() {
  local extra=(); [ "$1" = 0 ] && extra=(--session-id "$SESSION") || extra=(--resume "$SESSION")
  ( cd "$WS" && env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u ANTHROPIC_API_KEY \
      ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_AUTH_TOKEN=proxy-authenticates \
      ANTHROPIC_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
      CLAUDE_CODE_SUBAGENT_MODEL="$MODEL" CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
      claude -p "$2" "${extra[@]}" --setting-sources project --dangerously-skip-permissions --output-format stream-json --verbose \
        --disable-slash-commands --strict-mcp-config --no-chrome < /dev/null )
  # --setting-sources project: ~/.claude/settings.json is NOT loaded — on this Mac cc-switch keeps an env block there
  # (ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic, deepseek-v4-pro) that overrides the process env; the first
  # smoke test (09-19) went to DeepSeek's own API with the user's key because of it
}
round=0; text="$PROMPT"
while :; do
  log "- round $round start ($([ $round = 0 ] && echo prompt || echo continue))"
  set +e
  if [ "$ARM" = codex ]; then run_codex "$round" "$text" >> "$WS/agent_events.jsonl" 2>> "$WS/agent_stderr.log"
  else run_claude "$round" "$text" >> "$WS/agent_events.jsonl" 2>> "$WS/agent_stderr.log"; fi
  rc=$?; set -e
  log "- round $round exit=$rc"
  if committed; then log "- submission has a commit: $(git -C "$WS/submission" log --oneline -1)"; break; fi
  round=$((round+1)); text="$CONTINUE"
  echo "$(date '+%F %T') continue #$round: no commit in submission/ yet" >> "$WS/interactions.log"
  [ $round -le "$MAX_CONT" ] || { log "- ⚠️ $MAX_CONT continues and still no commit; stopping"; break; }
done

# ---- 5. audit ----
python3 - "$WS" "$ARM" "$MODEL" > "$WS/AUDIT.txt" <<'PY'
import json, sys
ws, arm, model = sys.argv[1:4]
rows = [json.loads(l) for l in open(f"{ws}/proxy_requests.log") if l.strip()]
bad = []
models = {r.get("model") for r in rows}
if models != {model}: bad.append(f"model(s) {models}")
if not all(r.get("injected") for r in rows): bad.append("a request without the thinking injection")
if not all((r.get("auth") or "").startswith("proxy:") for r in rows): bad.append("a request not authenticated by the proxy")
key = "thinking_blocks" if arm == "claude" else "reasoning_items"
over = [r for r in rows if ((r.get("usage") or {}).get("reasoning_tokens") or 0) > 0 or (r.get(key) or 0) > 0]
if over: bad.append(f"{len(over)} responses with reasoning ({key} / reasoning_tokens > 0)")
errs = [r for r in rows if r.get("status") != 200]
tok = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in rows), sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in rows)
print(f"requests {len(rows)}  models {sorted(m or '?' for m in models)}  non-200 {len(errs)}  prompt/completion tokens {tok[0]}/{tok[1]}")
print("CALIBER_OK" if not bad else "CALIBER_BROKEN: " + "; ".join(bad))
PY
cat "$WS/AUDIT.txt" | tee -a "$WS/RUN_NOTES.md"
BL="$(grep -vE '^\s*(#|$)' "$WS/paper/blacklist.txt" | head -1 | sed 's#https\?://github.com/##; s#/$##')"
if [ -n "$BL" ]; then
  hits="$(grep -rIl "$BL" "$WS/submission" 2>/dev/null | grep -v '/\.git/' || true)"
  [ -z "$hits" ] && log "- blacklist ($BL): no mention in the submission" || log "- ⚠️ blacklist ($BL) mentioned in: $(echo "$hits" | tr '\n' ' ') — check it is a citation, not copied code"
fi
n_files=$(git -C "$WS/submission" ls-files | wc -l | tr -d ' '); n_py=$(git -C "$WS/submission" ls-files '*.py' | wc -l | tr -d ' ')
untracked=$(git -C "$WS/submission" status --porcelain | wc -l | tr -d ' ')
log "- submission: $n_files tracked files ($n_py .py), $untracked uncommitted/untracked (git clean -fd drops them at grading)"

# ---- 6. into the pool ----
if [ "$TO_POOL" = 1 ] && committed; then
  n=1; while [ -e "$POOL/$PAPER/$ARM$n" ]; do n=$((n+1)); done
  mkdir -p "$POOL/$PAPER" && cp -R "$WS/submission" "$POOL/$PAPER/$ARM$n"
  log "- copied to $POOL/$PAPER/$ARM$n (grade: PAPER=$PAPER PB_JUDGE_MODEL=DeepSeek-V4-Flash bash deepcode_test/scripts/run_grade.sh)"
else
  log "- not copied to the pool ($([ "$TO_POOL" = 1 ] && echo 'no commit' || echo '--no-pool'))"
fi
log "done"
