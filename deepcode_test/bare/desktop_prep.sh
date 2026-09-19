#!/usr/bin/env bash
# Desktop arm, step 1 of 2: build the workspace + PROMPT.txt, start the thinking-off proxy for THIS run, put the prompt on
# the clipboard, print the checklist for the app. Then you drive the desktop app; finish with desktop_finish.sh.
#
#   bash desktop_prep.sh <codex|claude> <paper-id> [--hours N]
#
# The desktop apps (Codex app, Claude desktop's Code tab) read their model route from their own config (cc-switch writes
# it), so once — not per run — create a cc-switch profile per app that points at this proxy:
#   Codex:          model_providers.custom.base_url = "http://127.0.0.1:8787/v1", any placeholder token, wire_api responses
#   Claude desktop: ANTHROPIC_BASE_URL = "http://127.0.0.1:8788/anthropic", any placeholder ANTHROPIC_AUTH_TOKEN
# The proxy pins the model (PROXY_FORCE_MODEL=deepseek-flash, whatever the app's picker says), injects thinking off, replaces
# the User-Agent, drops x-codex-* headers, authenticates upstream with DEEPSEEK_API_KEY from ENV_FILE, and logs every request.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"; shift 2
HOURS=""; while [ $# -gt 0 ]; do case "$1" in --hours) HOURS="$2"; shift 2 ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
[ "$ARM" = codex ] || [ "$ARM" = claude ] || { echo "arm must be codex or claude"; exit 2; }
ENV_FILE="${ENV_FILE:-$HOME/Documents/env/deepseek.env}"; MODEL="${BARE_MODEL:-deepseek-flash}"; UPSTREAM="${BARE_UPSTREAM:-https://api.deepseek.com}"
[ -f "$ENV_FILE" ] && grep -q '^DEEPSEEK_API_KEY=.\+' "$ENV_FILE" || { echo "❌ $ENV_FILE missing or has no DEEPSEEK_API_KEY="; exit 1; }
PORT=$([ "$ARM" = codex ] && echo 8787 || echo 8788)
ROOT="${BARE_ROOT:-$REPO/work}"; WS="$ROOT/$PAPER-$ARM-desktop"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "❌ port $PORT busy — a previous run's proxy? finish it first (desktop_finish.sh) or kill it"; exit 1; fi

mkdir -p "$ROOT"; RLOG="$(mktemp)"
bash "$HERE/render_prompt.sh" "$PAPER" "$WS" ${HOURS:+--hours "$HOURS"} | tee "$RLOG"; grep -q PROMPT_OK "$RLOG" || exit 1; mv "$RLOG" "$WS/render.log"
{ echo "# $PAPER / $ARM desktop — $(date '+%F %T')"
  echo "- model: $MODEL via proxy :$PORT → $UPSTREAM (thinking off, model pinned by the proxy)"
  echo "- validation repo: $(git -C "$HERE" rev-parse --short HEAD)"
  echo "- time limit in prompt: ${HOURS:-none (no_time_limit_template)}"
  echo "- app version: (fill in: Codex app / Claude desktop 'About')"
  echo "- plugins / skills / MCP left on: (fill in, ideally 'none')"
  echo "- approval mode: (fill in: full auto)"
} > "$WS/RUN_NOTES.md"
( set -a; . "$ENV_FILE"; set +a
  PROXY_THINKING=disabled PROXY_UPSTREAM="$UPSTREAM" PROXY_USER_AGENT="paperbench-bare/1" PROXY_UPSTREAM_KEY_ENV=DEEPSEEK_API_KEY PROXY_FORCE_MODEL="$MODEL" \
  exec python3 "$HERE/paratera_proxy.py" "$PORT" "$WS/proxy_requests.log" > "$WS/proxy_stdout.log" 2>&1 ) &
echo $! > "$WS/proxy.pid"; sleep 1; kill -0 "$(cat "$WS/proxy.pid")" 2>/dev/null || { echo "❌ proxy did not start"; cat "$WS/proxy_stdout.log"; exit 1; }
echo "$(date '+%F %T') prep: proxy pid $(cat "$WS/proxy.pid") on :$PORT" >> "$WS/RUN_NOTES.md"
cat "$WS/PROMPT.txt" | pbcopy 2>/dev/null && CLIP="(already on the clipboard)" || CLIP=""
cat <<TXT

READY  $WS
  proxy :$PORT → $UPSTREAM, model pinned to $MODEL, thinking off; log $WS/proxy_requests.log

Now in the $([ "$ARM" = codex ] && echo "Codex app" || echo "Claude desktop app (Code tab)"):
  1. cc-switch: the profile that points at http://127.0.0.1:$PORT$([ "$ARM" = codex ] && echo "/v1" || echo "/anthropic") must be current; restart the app after switching.
  2. Open the folder  $WS  as the project. Approval: full auto. Plugins / browser / computer-use / skills / MCP: off.
     $([ "$ARM" = codex ] && echo "~/.codex/AGENTS.md must be empty; no AGENTS.md in the folder." || echo "No ~/.claude/CLAUDE.md, no CLAUDE.md in the folder; memory for a new folder is empty.")
  3. Paste PROMPT.txt as the first message, verbatim $CLIP — nothing before or after it.
  4. If it stops and asks, or stops without committing: reply with CONTINUE.txt verbatim (max 5 times) and add a line to
     $WS/interactions.log  (time, what it asked). Never answer a question with information.
  5. When it says it is done and submission/ has a commit:  bash $HERE/desktop_finish.sh $ARM $PAPER
TXT
