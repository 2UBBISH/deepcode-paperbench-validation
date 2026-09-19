#!/usr/bin/env bash
# One bare-arm run, start to finish: workspace + PROMPT.txt → proxy (thinking off, key from the env file) → the CLI
# non-interactively → continue rounds while nothing is committed → audit → copy into the judge's pool.
#
#   bash run_bare.sh <codex|claude> <paper-id> [--hours N] [--root DIR] [--pool DIR] [--max-continues N] [--no-pool]
#
# Caliber (docs/CODEDEV-ARMS.md §4): deepseek-flash on api.deepseek.com, thinking OFF, PaperBench's official Code-Dev
# instructions + its ADDITIONAL NOTES, paper.md + addendum + blacklist as the only input (same bytes as the DeepCode arm),
# no rubric, blacklist audited afterwards. The only secret is DEEPSEEK_API_KEY in ENV_FILE (default
# ~/Documents/env/deepseek.env): the local proxy sends it upstream; the CLIs get a placeholder token and a provider defined
# on the command line / in env vars, so nothing in ~/.codex or ~/.claude is read or changed for the model route.
# Thinking off needs the proxy — neither CLI can do it alone against DeepSeek (measured 2026-09-19):
#   · Claude Code's switches (CLAUDE_CODE_DISABLE_THINKING, MAX_THINKING_TOKENS=0) only OMIT the thinking field, and
#     DeepSeek's Anthropic endpoint then defaults to thinking on; the off switch is an explicit thinking:{type:disabled}.
#   · Codex sends reasoning.effort (the documented Responses-API switch, none = off), but DeepSeek serves a Codex
#     profile keyed on the User-Agent codex_exec/… or the x-codex-turn-metadata header that ignores effort=none.
# So the proxy injects (thinking disabled / effort none), replaces the User-Agent, drops x-codex-* headers, and its log
# proves the result: reasoning_tokens + reasoning_items (Codex) or thinking_blocks (Claude Code) are 0 on every line
# (AUDIT.txt → CALIBER_OK). Claude Code runs with --setting-sources project so a user-level settings.json cannot
# redirect it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"; shift 2
REPO="$(cd "$HERE/../.." && pwd)"
HOURS=""; ROOT="${BARE_ROOT:-$REPO/work}"; POOL="${RESULTS_ROOT:-$REPO/results}"; MAX_CONT=5; TO_POOL=1
ENV_FILE="${ENV_FILE:-$HOME/Documents/env/deepseek.env}"   # DEEPSEEK_API_KEY=… ; the proxy authenticates upstream with it
while [ $# -gt 0 ]; do case "$1" in
  --hours) HOURS="$2"; shift 2 ;; --root) ROOT="$2"; shift 2 ;; --pool) POOL="$2"; shift 2 ;;
  --max-continues) MAX_CONT="$2"; shift 2 ;; --no-pool) TO_POOL=0; shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
[ "$ARM" = codex ] || [ "$ARM" = claude ] || { echo "arm must be codex or claude"; exit 2; }
export PATH="$HOME/.local/node-v24.21.0/bin:$PATH"   # where npm -g put codex / claude on this Mac
command -v "$ARM" >/dev/null || { echo "❌ $ARM CLI not on PATH"; exit 1; }
MODEL="${BARE_MODEL:-deepseek-flash}"                 # DeepSeek's own id for V4 Flash
UPSTREAM="${BARE_UPSTREAM:-https://api.deepseek.com}"
UA="paperbench-bare/1"
[ -f "$ENV_FILE" ] || { echo "❌ $ENV_FILE missing — one line: DEEPSEEK_API_KEY=…"; exit 1; }
grep -q '^DEEPSEEK_API_KEY=.\+' "$ENV_FILE" || { echo "❌ $ENV_FILE has no DEEPSEEK_API_KEY="; exit 1; }
PORT=$([ "$ARM" = codex ] && echo 8787 || echo 8788)
WS="$ROOT/$PAPER-$ARM"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WS/RUN_NOTES.md"; }

# ---- 1. workspace + prompt (refuses a non-empty directory: one workspace per run) ----
mkdir -p "$ROOT"; RLOG="$(mktemp)"   # render_prompt.sh refuses a non-empty workspace, so its log lands there afterwards
bash "$HERE/render_prompt.sh" "$PAPER" "$WS" ${HOURS:+--hours "$HOURS"} | tee "$RLOG"
grep -q PROMPT_OK "$RLOG" || { echo "❌ prompt not rendered"; exit 1; }
mv "$RLOG" "$WS/render.log"
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
( set -a; . "$ENV_FILE"; set +a   # the key lives only in this subshell; the proxy sends it upstream, the CLIs never see it
  PROXY_THINKING=disabled PROXY_UPSTREAM="$UPSTREAM" PROXY_USER_AGENT="$UA" PROXY_UPSTREAM_KEY_ENV=DEEPSEEK_API_KEY \
  exec python3 "$HERE/paratera_proxy.py" "$PORT" "$WS/proxy_requests.log" > "$WS/proxy_stdout.log" 2>&1 ) &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
sleep 1; kill -0 $PROXY_PID 2>/dev/null || { echo "❌ proxy did not start (port $PORT busy?)"; cat "$WS/proxy_stdout.log"; exit 1; }
log "- proxy pid $PROXY_PID on :$PORT → $UPSTREAM (thinking off injected, UA $UA, key from $ENV_FILE)"

# ---- 4. the run, then continue rounds while nothing is committed ----
PROMPT="$(cat "$WS/PROMPT.txt")"
CONTINUE="$(cat "$WS/CONTINUE.txt")"
SESSION="$(python3 -c 'import uuid; print(uuid.uuid4())')"
committed() { git -C "$WS/submission" rev-parse --verify HEAD >/dev/null 2>&1; }
run_codex() {  # $1 = round (0 = first), $2 = prompt
  # a provider defined entirely on the command line: ~/.codex/config.toml is not touched and its own providers/keys are not used
  local common=(-c "model=$MODEL" -c model_provider=bare -c model_providers.bare.name=bare -c "model_providers.bare.base_url=http://127.0.0.1:$PORT/v1"
                -c model_providers.bare.wire_api=responses -c model_providers.bare.requires_openai_auth=false
                -c model_providers.bare.experimental_bearer_token=proxy-authenticates
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
      ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT/anthropic" ANTHROPIC_AUTH_TOKEN=proxy-authenticates \
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
python3 "$HERE/audit.py" "$WS" "$ARM" "$MODEL" > "$WS/AUDIT.txt" || true
cat "$WS/AUDIT.txt" | tee -a "$WS/RUN_NOTES.md"
BL="$(grep -vE '^\s*(#|$)' "$WS/paper/blacklist.txt" | head -1 | sed -e 's#https://github.com/##' -e 's#http://github.com/##' -e 's#/$##' -e 's#\.git$##')"
if [ -n "$BL" ]; then
  hits="$(grep -rIl "$BL" "$WS/submission" 2>/dev/null | grep -v '/\.git/' || true)"
  [ -z "$hits" ] && log "- blacklist ($BL): no mention in the submission" || log "- ⚠️ blacklist ($BL) mentioned in: $(echo "$hits" | tr '\n' ' ') — check it is a citation, not copied code"
fi
n_files=$(git -C "$WS/submission" ls-files | wc -l | tr -d ' '); n_py=$(git -C "$WS/submission" ls-files '*.py' | wc -l | tr -d ' ')
untracked=$(git -C "$WS/submission" status --porcelain | wc -l | tr -d ' ')
log "- submission: $n_files tracked files ($n_py .py), $untracked uncommitted/untracked (git clean -fd drops them at grading)"

# ---- 6. into the pool ----
if [ "$TO_POOL" = 1 ] && committed && [ -n "${RESULTS_ROOT:-}" ]; then
  # collaboration layout: RESULTS_ROOT/<paper>/<arm>/{submission/, RUN_NOTES.md, AUDIT.txt, proxy_requests.log, agent_events.jsonl, …}
  DEST="$POOL/$PAPER/$ARM"; rm -rf "$DEST"; mkdir -p "$DEST"
  cp -R "$WS/submission" "$DEST/submission"
  for f in RUN_NOTES.md AUDIT.txt proxy_requests.log agent_events.jsonl agent_stderr.log interactions.log PROMPT.txt render.log; do [ -f "$WS/$f" ] && cp "$WS/$f" "$DEST/$f"; done
  cp "$WS"/last_message.*.txt "$DEST/" 2>/dev/null || true
  log "- results in $DEST (submission/ + run records)"
elif [ "$TO_POOL" = 1 ] && committed; then
  n=1; while [ -e "$POOL/$PAPER/$ARM$n" ]; do n=$((n+1)); done
  mkdir -p "$POOL/$PAPER" && cp -R "$WS/submission" "$POOL/$PAPER/$ARM$n"
  log "- copied to $POOL/$PAPER/$ARM$n (grade: PAPER=$PAPER PB_JUDGE_MODEL=DeepSeek-V4-Flash bash deepcode_test/scripts/run_grade.sh)"
else
  log "- not copied to the pool ($([ "$TO_POOL" = 1 ] && echo 'no commit' || echo '--no-pool'))"
fi
log "done"
