#!/usr/bin/env bash
# Build one paper's bare-run (Codex) workspace and its PROMPT.txt.
#
#   bash render_prompt.sh <paper-id> <workspace-dir> [--hours N]
#
# PROMPT.txt = PaperBench's official Code-Dev instructions (code_only_instructions.txt, byte-identical source)
#            + PaperBench's own ADDITIONAL NOTES block (additional_notes.txt = paperbench/solvers/basicagent/prompts/
#              templates.py `additional_notes_template`, filled the way the benchmark fills it: no GPU on this Mac,
#              `no_time_limit_template` by default or `time_limit_template` with --hours N)
#            with /home/paper and /home/submission replaced by this workspace's paths.
# The workspace gets exactly what the benchmark gives an agent: paper.pdf, paper.md, addendum.md, blacklist.txt,
# assets/ (never rubric.json / config.yaml), an empty agent.env (the notes name it), an empty submission/ with git init.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PB="$REPO/frontier-evals/project/paperbench"
OFFICIAL="$PB/paperbench/instructions/code_only_instructions.txt"
PAPER="${1:?paper id}"; WS="${2:?workspace dir}"; shift 2
HOURS=""
while [ $# -gt 0 ]; do case "$1" in --hours) HOURS="$2"; shift 2 ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
SRC="$PB/data/papers/$PAPER"
[ -f "$SRC/paper.md" ] || { echo "❌ $SRC/paper.md missing"; exit 1; }
[ -f "$OFFICIAL" ] || { echo "❌ $OFFICIAL missing (judge.sh setup first)"; exit 1; }
WS="$(mkdir -p "$WS" && cd "$WS" && pwd)"
[ -z "$(ls -A "$WS")" ] || { echo "❌ $WS is not empty; a workspace is used once"; exit 1; }

mkdir -p "$WS/paper" "$WS/submission"
for f in paper.pdf paper.md addendum.md blacklist.txt; do [ -f "$SRC/$f" ] && cp "$SRC/$f" "$WS/paper/$f"; done
[ -d "$SRC/assets" ] && cp -R "$SRC/assets" "$WS/paper/assets"
if grep -rlq "^version https://git-lfs" "$WS/paper/assets" 2>/dev/null; then
  echo "  ⚠️ assets are LFS pointers (Codex can read them but not view the figures); hydrate from HF if figures should count"
fi
: > "$WS/agent.env"
git -C "$WS/submission" init -q
ls "$WS/paper" | grep -Eq 'rubric|config' && { echo "❌ rubric/config leaked into the workspace"; exit 1; }

if [ -n "$HOURS" ]; then
  TL="You are expected to work until you have spent $HOURS hours on this task. You may iteratively develop your solution within these constraints, but bear in mind that program runtime also counts towards the time limit. This is a difficult task and we expect a good solution to require using the full time limit."
else
  TL="You are expected to work until you have reproduced all the core contributions of the paper."
fi
{ cat "$OFFICIAL"; sed -e "s|{TIME_LIMIT}|$TL|" -e "s|{WORKSPACE}|$WS|g" "$HERE/additional_notes.txt"; } \
  | sed -e "s#/home/paper#$WS/paper#g" -e "s#/home/submission#$WS/submission#g" > "$WS/PROMPT.txt"
cp "$HERE/continue_message.txt" "$WS/CONTINUE.txt"
: > "$WS/interactions.log"

# proof: the official part of PROMPT.txt is the official file with only the two path substitutions
head -n "$(wc -l < "$OFFICIAL")" "$WS/PROMPT.txt" | diff - <(sed -e "s#/home/paper#$WS/paper#g" -e "s#/home/submission#$WS/submission#g" "$OFFICIAL") \
  && echo "PROMPT_OK  $WS/PROMPT.txt  ($(wc -l < "$WS/PROMPT.txt") lines; time limit: ${HOURS:-none})"
for f in paper.pdf paper.md addendum.md blacklist.txt; do [ -f "$SRC/$f" ] && shasum -a 256 "$SRC/$f" "$WS/paper/$f" | cut -c1-12 | uniq -c | awk -v f=$f '{print "  "$0"  "f}'; done
echo "copy the prompt:  cat $WS/PROMPT.txt | pbcopy"
