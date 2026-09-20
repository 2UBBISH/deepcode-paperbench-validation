#!/usr/bin/env bash
# Desktop arm, step 1 of 2: build the workspace + PROMPT.txt, record the start time, put the prompt on the clipboard,
# print the checklist. Then you drive the desktop app; finish with desktop_finish.sh.
#
#   bash desktop_prep.sh <codex|claude> <paper-id> [--hours N]      (default 3: the official time_limit_template sentence)
#
# Caliber of this batch: deepseek-flash @ api.deepseek.com, thinking ON (DeepSeek's default — nothing to inject, so no proxy;
# the apps talk to DeepSeek through the cc-switch profile you set once: Codex → provider base_url https://api.deepseek.com/v1,
# model deepseek-flash; Claude desktop → ANTHROPIC_BASE_URL https://api.deepseek.com/anthropic and ANTHROPIC_MODEL +
# ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL + CLAUDE_CODE_SUBAGENT_MODEL = deepseek-flash). Evidence comes from the apps'
# own session logs (desktop_finish.sh → audit_desktop.py: model on every turn, thinking tokens > 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
ARM="${1:?codex|claude}"; PAPER="${2:?paper id}"; shift 2
HOURS="${BARE_HOURS:-3}"; while [ $# -gt 0 ]; do case "$1" in --hours) HOURS="$2"; shift 2 ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
[ "$ARM" = codex ] || [ "$ARM" = claude ] || { echo "arm must be codex or claude"; exit 2; }
ROOT="${BARE_ROOT:-$REPO/work}"; WS="$ROOT/$PAPER-$ARM-desktop"
if [ "$ARM" = codex ]; then [ ! -s "$HOME/.codex/AGENTS.md" ] || { echo "❌ ~/.codex/AGENTS.md is not empty"; exit 1; }
else [ ! -f "$HOME/.claude/CLAUDE.md" ] || { echo "❌ ~/.claude/CLAUDE.md exists; move it aside for the run"; exit 1; }; fi

# execution (owner 09-20 evening): the agent runs normally in full-auto mode and may run commands; the prompt's Execution
# note says only long CPU / GPU training or evaluation is out (the code runs remotely later). Nothing is blocked or gated;
# audit_desktop.py lists what ran with wall time and a run with a long experiment (> 10 min one command / > 60 min total) is void.
# Earlier rules files from the 09-20 daytime variants are removed so they do not interfere.
if [ "$ARM" = codex ]; then
  for old in paperbench-no-exec.rules paperbench-exec.rules; do
    [ -f "$HOME/.codex/rules/$old" ] && rm -f "$HOME/.codex/rules/$old" && echo "  removed the old ~/.codex/rules/$old"
  done
fi
mkdir -p "$ROOT"; RLOG="$(mktemp)"
bash "$HERE/render_prompt.sh" "$PAPER" "$WS" ${HOURS:+--hours "$HOURS"} | tee "$RLOG"; grep -q PROMPT_OK "$RLOG" || exit 1; mv "$RLOG" "$WS/render.log"
START=$(date +%s); echo "$START" > "$WS/START_EPOCH"
{ echo "# $PAPER / $ARM desktop — $(date '+%F %T')"
  echo "- caliber: deepseek-flash @ api.deepseek.com via the app's cc-switch profile, thinking ON (DeepSeek default), execution = commands allowed, only long CPU / GPU training or evaluation is out (the prompt says the code runs remotely later); full auto, nothing gated"
  echo "- validation repo: $(git -C "$HERE" rev-parse --short HEAD)"
  echo "- time budget told to the agent: $HOURS h (official time_limit_template sentence); not a hard cap — the agent stops when it believes the core contributions are reproduced, nobody kills it at $HOURS h"
  echo "- app version: (fill in: Codex app / Claude desktop 'About')"
  echo "- plugins / skills / MCP left on: (fill in, ideally 'none')"
  echo "- approval mode: (fill in: full auto)"
} > "$WS/RUN_NOTES.md"
: > "$WS/interactions.log"
cat "$WS/PROMPT.txt" | pbcopy 2>/dev/null && CLIP="(already on the clipboard)" || CLIP=""
cat <<TXT

READY  $WS      started $(date '+%H:%M'); the prompt tells the agent it has $HOURS h (official sentence, not a cap — do not stop it at $HOURS h)

In the $([ "$ARM" = codex ] && echo "Codex app" || echo "Claude desktop app (Code tab)"):
  1. cc-switch: the DeepSeek profile (deepseek-flash) must be current; restart the app after switching.
  2. Open the folder  $WS  as the project. Approval: full auto. Plugins / browser / computer-use / skills / MCP: off.
     It may run commands (installs, checks, tests, short scripts); the prompt says only long CPU / GPU training or
     evaluation is out (the code runs remotely later). If you see a training / evaluation grinding on for many
     minutes, stop that command (not the session) and note it in interactions.log; desktop_finish.sh lists what ran.
  3. Paste PROMPT.txt as the first message, verbatim $CLIP — nothing before or after it.
  4. If it stops and asks, or stops without committing: reply with CONTINUE.txt verbatim (max 5 times) and add a line to
     $WS/interactions.log  (time, what it asked). Never answer a question with information.
  5. When it says it is done and submission/ has a commit (let it run past $HOURS h if it is still working —
     the $HOURS h is what the official prompt tells it, not a cutoff; the elapsed time is recorded):
        bash $HERE/desktop_finish.sh $ARM $PAPER
TXT
