#!/usr/bin/env bash
# Desktop arm, step 2 of 2: stop this run's proxy, audit the caliber from its log, check the blacklist and the submission,
# copy everything into results/<paper>/<arm>/.      bash desktop_finish.sh <codex|claude> <paper-id>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"
ROOT="${BARE_ROOT:-$REPO/work}"; WS="$ROOT/$PAPER-$ARM-desktop"; RES="${RESULTS_ROOT:-$REPO/results}/$PAPER/$ARM"
MODEL="${BARE_MODEL:-deepseek-flash}"
[ -d "$WS" ] || { echo "❌ no workspace $WS (desktop_prep.sh first)"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WS/RUN_NOTES.md"; }
if [ -f "$WS/proxy.pid" ]; then kill "$(cat "$WS/proxy.pid")" 2>/dev/null && log "- proxy stopped" || true; rm -f "$WS/proxy.pid"; fi
python3 "$HERE/audit.py" "$WS" "$ARM" "$MODEL" > "$WS/AUDIT.txt" || true
cat "$WS/AUDIT.txt" | tee -a "$WS/RUN_NOTES.md"
if ! git -C "$WS/submission" rev-parse --verify HEAD >/dev/null 2>&1; then
  log "- ⚠️ submission/ has no commit; committing the working tree as-is for the record"
  git -C "$WS/submission" add -A && git -C "$WS/submission" -c user.name=bare -c user.email=bare@local commit -qm "final state (committed by desktop_finish.sh)" || true
fi
BL="$(grep -vE '^\s*(#|$)' "$WS/paper/blacklist.txt" | head -1 | sed -e 's#https://github.com/##' -e 's#http://github.com/##' -e 's#/$##' -e 's#\.git$##')"
if [ -n "$BL" ] && [ "$BL" != none ]; then
  hits="$(grep -rIl "$BL" "$WS/submission" 2>/dev/null | grep -v '/\.git/' || true)"
  [ -z "$hits" ] && log "- blacklist ($BL): no mention in the submission" || log "- ⚠️ blacklist ($BL) mentioned in: $(echo "$hits" | tr '\n' ' ') — check it is a citation, not copied code"
fi
n_files=$(git -C "$WS/submission" ls-files | wc -l | tr -d ' '); n_py=$(git -C "$WS/submission" ls-files '*.py' | wc -l | tr -d ' ')
log "- submission: $n_files tracked files ($n_py .py); continues: $(grep -c . "$WS/interactions.log" 2>/dev/null || echo 0)"
rm -rf "$RES"; mkdir -p "$RES"; cp -R "$WS/submission" "$RES/submission"
for f in RUN_NOTES.md AUDIT.txt proxy_requests.log interactions.log PROMPT.txt render.log; do [ -f "$WS/$f" ] && cp "$WS/$f" "$RES/$f"; done
log "- results in $RES"
