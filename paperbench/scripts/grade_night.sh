#!/bin/bash
# Night grading v4 (03:55 09-22). run_grade.sh's [4/4] copies grade.json from the NEWEST nanoeval run group of ANY paper
# (ls -t runs/ | head -1) — right when one grading runs at a time, wrong with four: a finished paper's [4/4] finds another
# paper's still-empty group and copies nothing (lca/codex, robust-clip/deepcode tonight). v4 ignores [4/4] and collects
# straight from the nanoeval run dirs (<paper>_<runid>/grade.json newer than a per-grading marker). run_grade.sh itself
# is left untouched while gradings run (bash reads scripts incrementally); fix + commit after the night.
export PATH=$HOME/Documents/search/.tools/bootstrap/bin:$PATH
V=~/Documents/env/paperbench-judge/validation/paperbench; T=~/Documents/0919-test; P=~/pb_submissions; A=~/pb_submissions_archive; LOG=$V/runs/grade_night_0923.log; NR=$V/frontier-evals/project/paperbench/runs
in_window() { local m=$((10#$(date +%H)*60+10#$(date +%M))); [ $m -ge 120 ] && [ $m -le $((8*60-50)) ]; }
window_wait() { while ! in_window; do local m=$((10#$(date +%H)*60+10#$(date +%M))); if [ $m -lt 120 ]; then sleep $(( (120-m)*60 )); else echo "$(date +%F\ %T) outside the window, sleeping until 02:00" >> $LOG; sleep $(( $(date -j -v+1d -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 02:00" +%s) - $(date +%s) )); fi; done; }
collect() {  # $1 paper $2 arm $3 marker : grade.json of this paper's nanoeval runs newer than the marker, not yet banked
  local paper=$1 arm=$2 marker=$3 n=0
  for g in $(find $NR -maxdepth 3 -path "*/${paper}_*/grade.json" -newer $marker 2>/dev/null); do
    local rid=$(basename $(dirname $g) | sed "s/^${paper}_\([0-9a-f]\{8\}\).*/\1/")
    ls $T/grades/${paper}_*_${rid}_* >/dev/null 2>&1 && continue
    n=$((n+1)); cp $g $T/grades/${paper}_${arm}_${rid}_sf_tree_thinkoff_pro.grade.json
    python3 -c "import json;d=json.load(open('$g'));jo=d['paperbench_result']['judge_output'];print('$(date +%H:%M) $paper/$arm: score', round(jo['score'],4), 'leaves', jo['num_leaf_nodes'], 'invalid', jo['num_invalid_leaf_nodes'])" >> $LOG
  done
  [ $n -eq 0 ] && echo "$(date +%H:%M) $paper/$arm: NO grade.json produced — see runs/$paper/grade_night_0923_$arm.log" >> $LOG
}
tidy() { mkdir -p $A/$1/night_0923_graded; mv $P/$1/$2 $A/$1/night_0923_graded/${2}_$(date +%H%M) 2>/dev/null; mv $A/$1/night_0923_staging/* $P/$1/ 2>/dev/null; }
grade_one() {  # $1 paper $2 arm → 0 done / 2 deferred
  local paper=$1 arm=$2
  ls $T/grades/${paper}_${arm}_*.grade.json >/dev/null 2>&1 && { echo "$(date +%H:%M) $paper/$arm: already graded, skipped" >> $LOG; return 0; }
  if [ $arm = deepcode ]; then [ -d $P/$paper/deepcode ] || [ -d $A/$paper/night_0923_staging/deepcode ] || { echo "$(date +%H:%M) $paper/deepcode: no tree yet, deferred" >> $LOG; return 2; }
  else [ -d $T/codex/results/$paper/codex/submission ] || { echo "$(date +%H:%M) $paper/codex: no tree yet, deferred" >> $LOG; return 2; }; fi
  mkdir -p $V/runs/$paper/grades $A/$paper/night_0923_staging $A/$paper/night_0923_graded
  mv $P/$paper/* $A/$paper/night_0923_staging/ 2>/dev/null; mkdir -p $P/$paper/$arm
  local src=$T/codex/results/$paper/codex/submission; [ $arm = deepcode ] && src=$A/$paper/night_0923_staging/deepcode
  rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache $src/ $P/$paper/$arm/
  local marker=$V/runs/$paper/.marker_$arm; touch $marker
  window_wait; echo "$(date +%H:%M) $paper/$arm: grading ($(find $P/$paper/$arm -name '*.py' | wc -l | tr -d ' ') py)" >> $LOG
  ( cd $V && PAPER=$paper bash scripts/run_grade.sh > $V/runs/$paper/grade_night_0923_$arm.log 2>&1 )
  collect $paper $arm $marker; tidy $paper $arm
}
adopt() {  # $1 paper $2 arm : a grading v3 left running under nanoeval — wait, collect, tidy
  while ps -eo command | grep -q "[p]aper_split=$1 "; do sleep 30; done
  collect $1 $2 $V/runs/grade_night_v2.pid; tidy $1 $2
}
worker() {
  local todo=() deferred=(); for paper in "$@"; do todo+=("$paper:codex" "$paper:deepcode"); done
  for t in "${todo[@]}"; do grade_one ${t%%:*} ${t##*:}; [ $? -eq 2 ] && deferred+=("$t"); done
  while [ ${#deferred[@]} -gt 0 ] && in_window; do
    local again=(); for t in "${deferred[@]}"; do grade_one ${t%%:*} ${t##*:}; [ $? -eq 2 ] && again+=("$t"); done
    deferred=("${again[@]}"); [ ${#deferred[@]} -gt 0 ] && sleep 300
  done
  [ ${#deferred[@]} -gt 0 ] && echo "$(date +%H:%M) worker left ungraded (no tree by window end): ${deferred[*]}" >> $LOG
}
echo "$(date +%F\ %T) v4 starts: collect from nanoeval run dirs; adopting lbcs/deepcode what-will/codex ftrl/codex pinn/codex lca/deepcode" >> $LOG
( adopt lbcs deepcode; worker lbcs sample-specific-masks stay-on-topic-with-classifier-free-guidance sequential-neural-score-estimation sapg ) &
( adopt what-will-my-model-forget codex; worker what-will-my-model-forget all-in-one bridging-data-gaps test-time-model-adaptation ) &
( adopt ftrl codex; adopt lca-on-the-line deepcode; worker ftrl lca-on-the-line adaptive-pruning mechanistic-understanding stochastic-interpolants ) &
( adopt pinn codex; worker pinn bam bbox robust-clip ) &
wait; echo "$(date +%F\ %T) night grading v4 done" >> $LOG
