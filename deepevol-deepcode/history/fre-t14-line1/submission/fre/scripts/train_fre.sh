#!/usr/bin/env bash
# =============================================================================
# train_fre.sh -- Launch FRE encoder (phase 1) + z-conditioned IQL policy
#                 (phase 2) training for one or more domains.
#
# Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings"
#        (ICML 2024), Algorithm 1 / Section 4.3 / Table 3.
#
# Strided training scheme reproduced here:
#   Phase 1 (encoder only): 150,000 steps  (1M for ExORL / Kitchen)
#                           -- FRE trained by maximizing Eq. (6) (MSE + beta KL)
#                           -- RL components are NOT trained
#   Phase 2 (policy):       850,000 steps  (1M for ExORL / Kitchen)
#                           -- encoder FROZEN so the mapping eta -> z stays
#                              stationary during TD learning
#                           -- pi(a|s,z), Q(s,a,z), V(s,z) trained with IQL
#                              using r = eta(s)
#
# Usage
# -----
#   bash fre/scripts/train_fre.sh                     # all domains, seeds 0..4
#   SEEDS="0" DOMAINS="antmaze" bash fre/scripts/train_fre.sh
#   DEVICE=cpu EPISODES=4 bash fre/scripts/train_fre.sh
#   EXTRA_ARGS="--prior-mixture all" bash fre/scripts/train_fre.sh
#   DRY_RUN=1 bash fre/scripts/train_fre.sh           # print the plan only
#
# Environment variables
# ---------------------
#   PYTHON            python interpreter (default: python)
#   DOMAINS           space separated domains (default: all four canonical names)
#   SEEDS             space separated seeds    (default: 0 1 2 3 4)
#   STAGE             encoder | policy | train | all  (default: all)
#   DEVICE            cpu | cuda | cuda:0      (default: "")
#   RUN_ROOT          output directory root    (default: runs)
#   DATASET_DIR       offline dataset root     (exported as FRE_DATASET_DIR)
#   DATASET_PATH      explicit dataset file    (single-domain runs only)
#   EPISODES          evaluation episodes per task (default: 20)
#   NO_PHYSICS        1 -> disable ExORL physics augmentation
#   EXTRA_ARGS        extra CLI args forwarded verbatim to fre/main.py
#   DRY_RUN           1 -> only print commands
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN="${REPO_ROOT}/fre/main.py"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-}"
RUN_ROOT="${RUN_ROOT:-runs}"
EPISODES="${EPISODES:-20}"
DRY_RUN="${DRY_RUN:-0}"
NO_PHYSICS="${NO_PHYSICS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Canonical domain names used throughout the codebase
# (fre/config/envs.py -> domain_names()); ExORL is split into walker/cheetah.
DOMAINS="${DOMAINS:-antmaze exorl_walker exorl_cheetah kitchen}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# Offline dataset location: the loader (fre/envs/d4rl_loader.py) reads
# FRE_DATASET_DIR (default "datasets") and falls back to the D4RL API.
if [[ -n "${DATASET_DIR:-}" ]]; then
  export FRE_DATASET_DIR="${DATASET_DIR}"
fi

if [[ ! -f "${MAIN}" ]]; then
  echo "error: cannot find ${MAIN}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Flag detection: fre/main.py is the single entry point and its argparse surface
# may evolve.  Rather than hard-failing on an unknown flag, we probe --help once
# and forward only the options the current parser actually accepts.
# -----------------------------------------------------------------------------
_HELP_TEXT=""
_load_help() {
  if [[ -z "${_HELP_TEXT}" ]]; then
    _HELP_TEXT="$("${PYTHON}" "${MAIN}" --help 2>&1 || true)"
  fi
}

_supports() {
  _load_help
  [[ "${_HELP_TEXT}" == *"$1"* ]]
}

CMD=()
_add_flag() {
  if _supports "$1"; then CMD+=("$1"); fi
}
_add_opt() {
  if _supports "$1"; then CMD+=("$1" "$2"); fi
}

build_command() {
  local domain="$1" seed="$2"
  CMD=("${PYTHON}" "${MAIN}")
  _add_opt --agent "fre"
  _add_opt --domain "${domain}"
  _add_opt --stage "${STAGE}"
  _add_opt --seed "${seed}"
  _add_opt --run-dir "${RUN_ROOT}"
  _add_opt --num-eval-episodes "${EPISODES}"
  if [[ -n "${DEVICE}" ]]; then _add_opt --device "${DEVICE}"; fi
  if [[ -n "${DATASET_PATH:-}" ]]; then _add_opt --dataset-path "${DATASET_PATH}"; fi
  if [[ "${NO_PHYSICS}" == "1" ]]; then _add_flag --no-physics; fi
  # Train only: skip the (expensive) zero-shot evaluation pass.
  _add_flag --no-eval
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(${EXTRA_ARGS})
  fi
}

quote_cmd() {
  local out="" a
  for a in "$@"; do out+="${a@Q} "; done
  printf '%s' "${out% }"
}

TOTAL=0
FAILED=0

echo "=============================================================="
echo " FRE training (Algorithm 1)"
echo "   repo           : ${REPO_ROOT}"
echo "   domains        : ${DOMAINS}"
echo "   seeds          : ${SEEDS}"
echo "   stage          : ${STAGE}   (encoder=150k/1M, policy=850k/1M)"
echo "   eval episodes  : ${EPISODES}"
echo "   dataset dir    : ${FRE_DATASET_DIR:-datasets}"
echo "   device         : ${DEVICE:-<default>}"
echo "   runs root      : ${RUN_ROOT}"
echo "--------------------------------------------------------------"

for domain in ${DOMAINS}; do
  for seed in ${SEEDS}; do
    build_command "${domain}" "${seed}"
    TOTAL=$((TOTAL + 1))
    cmd_str="$(quote_cmd ${CMD[@]+"${CMD[@]}"})"
    echo "[train_fre] ${domain} seed=${seed}"
    echo "            ${cmd_str}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi
    if ! "${CMD[@]}"; then
      FAILED=$((FAILED + 1))
      echo "[train_fre] WARNING: ${domain} seed=${seed} failed" >&2
    fi
  done
done

echo "--------------------------------------------------------------"
echo " finished: ${TOTAL} run(s), ${FAILED} failure(s)"

# -----------------------------------------------------------------------------
# Summary: render the produced rows next to the published Table 1 numbers.
# Kept separate from training so a partial run still prints what it has.
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" != "1" && "${FAILED}" == "0" ]]; then
  "${PYTHON}" - "${REPO_ROOT}" "${RUN_ROOT}" <<'PY'
import json
import os
import sys

repo_root, run_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)

try:
    from fre.utils.normalization import (
        TABLE1_AGGREGATE_TARGETS,
        TABLE1_FRE_TARGETS,
        format_mean_std,
        aggregate_task_scores,
    )
except Exception as exc:  # pragma: no cover - reporting is best effort
    print(f"[train_fre] aggregation skipped ({exc})")
    raise SystemExit(0)

# Collect the per-seed report.json files written by fre/main.py.
per_task = {}
scanned = 0
for root, _dirs, files in os.walk(run_root):
    if "report.json" not in files:
        continue
    path = os.path.join(root, "report.json")
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception:
        continue
    scanned += 1
    rows = payload.get("results") or payload.get("rows") or {}
    if isinstance(rows, dict):
        for name, value in rows.items():
            if isinstance(value, dict):
                score = value.get("mean", value.get("score"))
            else:
                score = value
            if score is None:
                continue
            per_task.setdefault(name, []).append(float(score))
    else:
        # List-of-dicts form: [{"task": name, "score": ...}, ...]
        for entry in rows:
            if isinstance(entry, dict) and "task" in entry:
                per_task.setdefault(entry["task"], []).append(
                    float(entry.get("score", entry.get("mean", 0.0)))
                )

print(f"\n[train_fre] aggregated {scanned} report file(s) under {run_root}/")

if not per_task:
    print("[train_fre] no per-task results found; nothing to summarise")
    raise SystemExit(0)

stats = aggregate_task_scores({task: scores for task, scores in per_task.items()})
print(f"\n{'task':<26}{'FRE (obs)':>18}{'FRE (paper)':>18}")
print("-" * 62)
for task in sorted(stats):
    obs = stats[task]
    ref = TABLE1_FRE_TARGETS.get(task)
    obs_str = format_mean_std(obs.get("mean", 0.0), obs.get("std", 0.0))
    ref_str = format_mean_std(*ref) if ref else "n/a"
    print(f"{task:<26}{obs_str:>18}{ref_str:>18}")

print("\naggregate targets (paper, Table 1):")
for row, (mean, std) in TABLE1_AGGREGATE_TARGETS.items():
    print(f"  {row:<12}{format_mean_std(mean, std)}")
PY
else
  echo "[train_fre] summary skipped (dry run or failures present)"
fi

echo "=============================================================="
exit 0
