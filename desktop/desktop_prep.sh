#!/usr/bin/env bash
# Desktop arm, step 1 of 2: build the workspace + PROMPT.txt, record the start time, put the prompt on the clipboard,
# print the checklist. Then you drive the desktop app; finish with desktop_finish.sh.
#
#   bash desktop_prep.sh <codex|claude> <paper-id> [--hours N]
#
# Caliber of this batch: deepseek-flash @ api.deepseek.com, thinking ON (DeepSeek's default — nothing to inject, so no proxy;
# the apps talk to DeepSeek through the cc-switch profile you set once: Codex → provider base_url https://api.deepseek.com/v1,
# model deepseek-flash; Claude desktop → ANTHROPIC_BASE_URL https://api.deepseek.com/anthropic and ANTHROPIC_MODEL +
# ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL + CLAUDE_CODE_SUBAGENT_MODEL = deepseek-flash). Evidence comes from the apps'
# own session logs (desktop_finish.sh → audit_desktop.py: model on every turn, thinking tokens > 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"; shift 2
HOURS=""; while [ $# -gt 0 ]; do case "$1" in --hours) HOURS="$2"; shift 2 ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
[ "$ARM" = codex ] || [ "$ARM" = claude ] || { echo "arm must be codex or claude"; exit 2; }
ROOT="${BARE_ROOT:-$REPO/work}"; WS="$ROOT/$PAPER-$ARM-desktop"
if [ "$ARM" = codex ]; then [ ! -s "$HOME/.codex/AGENTS.md" ] || { echo "❌ ~/.codex/AGENTS.md is not empty"; exit 1; }
else [ ! -f "$HOME/.claude/CLAUDE.md" ] || { echo "❌ ~/.claude/CLAUDE.md exists; move it aside for the run"; exit 1; }; fi

mkdir -p "$ROOT"; RLOG="$(mktemp)"
bash "$HERE/render_prompt.sh" "$PAPER" "$WS" ${HOURS:+--hours "$HOURS"} | tee "$RLOG"; grep -q PROMPT_OK "$RLOG" || exit 1; mv "$RLOG" "$WS/render.log"
START=$(date +%s); echo "$START" > "$WS/START_EPOCH"
{ echo "# $PAPER / $ARM desktop — $(date '+%F %T')"
  echo "- caliber: deepseek-flash @ api.deepseek.com via the app's cc-switch profile, thinking ON (DeepSeek default)"
  echo "- validation repo: $(git -C "$HERE" rev-parse --short HEAD)"
  echo "- time limit in prompt: ${HOURS:-none (no_time_limit_template)}"
  echo "- app version: (fill in: Codex app / Claude desktop 'About')"
  echo "- plugins / skills / MCP left on: (fill in, ideally 'none')"
  echo "- approval mode: (fill in: full auto)"
} > "$WS/RUN_NOTES.md"
: > "$WS/interactions.log"
cat "$WS/PROMPT.txt" | pbcopy 2>/dev/null && CLIP="(already on the clipboard)" || CLIP=""
cat <<TXT

READY  $WS      started $(date '+%H:%M')

In the $([ "$ARM" = codex ] && echo "Codex app" || echo "Claude desktop app (Code tab)"):
  1. cc-switch: the DeepSeek profile (deepseek-flash) must be current; restart the app after switching.
  2. Open the folder  $WS  as the project. Approval: full auto. Plugins / browser / computer-use / skills / MCP: off.
  3. Paste PROMPT.txt as the first message, verbatim $CLIP — nothing before or after it.
  4. If it stops and asks, or stops without committing: reply with CONTINUE.txt verbatim (max 5 times) and add a line to
     $WS/interactions.log  (time, what it asked). Never answer a question with information.
  5. When it says it is done and submission/ has a commit:
        bash $HERE/desktop_finish.sh $ARM $PAPER
TXT
