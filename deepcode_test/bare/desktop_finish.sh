#!/usr/bin/env bash
# Desktop arm, step 2 of 2: audit the caliber from the app's own session logs, check the cap, the blacklist and the
# submission, copy everything (incl. the matched session logs) into results/<paper>/<arm>/.
#     bash desktop_finish.sh <codex|claude> <paper-id>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"
ROOT="${BARE_ROOT:-$REPO/work}"; WS="$ROOT/$PAPER-$ARM-desktop"; RES="${RESULTS_ROOT:-$REPO/results}/$PAPER/$ARM"
MODEL="${BARE_MODEL:-deepseek-flash}"; CAP_H="${BARE_CAP_HOURS:-3}"
[ -d "$WS" ] || { echo "❌ no workspace $WS (desktop_prep.sh first)"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$WS/RUN_NOTES.md"; }
START=$(cat "$WS/START_EPOCH" 2>/dev/null || echo 0); ELAPSED=$(( ( $(date +%s) - START ) / 60 ))
log "- finished after $ELAPSED min$([ $ELAPSED -gt $((CAP_H*60)) ] && echo " — OVER the $CAP_H h cap (stopped at the cap: whatever is in submission/ counts)")"
python3 "$HERE/audit_desktop.py" "$ARM" "$WS" "$START" "$MODEL" --copy "$WS/session_logs" > "$WS/AUDIT.txt" || true
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
for f in RUN_NOTES.md AUDIT.txt interactions.log PROMPT.txt render.log START_EPOCH; do [ -f "$WS/$f" ] && cp "$WS/$f" "$RES/$f"; done
[ -d "$WS/session_logs" ] && cp -R "$WS/session_logs" "$RES/session_logs"
log "- results in $RES"
