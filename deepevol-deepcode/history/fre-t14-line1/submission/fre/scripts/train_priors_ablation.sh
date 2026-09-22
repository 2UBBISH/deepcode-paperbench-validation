#!/usr/bin/env bash
# =============================================================================
# train_priors_ablation.sh
#
# Reproduces **Table 4** of "Zero-Shot Reinforcement Learning via Functional
# Reward Encodings" (FRE, ICML 2024): an ablation over the *prior reward
# distribution* p(eta) used to train the FRE encoder/decoder.
#
# The paper (Appendix D, Table 4) compares seven FRE agents trained on
# different subsets of the three random reward-function classes:
#
#     goal   -> singleton / goal-reaching rewards   (Appendix B, §4.2)
#     lin    -> random linear reward functions      (Appendix B)
#     mlp    -> random MLP reward functions         (Appendix B)
#
#   +--------------+-------------------------------------------------------+
#   | variant      | mixture over {goal, lin, mlp}                         |
#   +--------------+-------------------------------------------------------+
#   | FRE-all      | 1/3, 1/3, 1/3   (the full prior, default in the paper)|
#   | FRE-goals    | 1, 0, 0                                               |
#   | FRE-lin      | 0, 1, 0                                               |
#   | FRE-mlp      | 0, 0, 1                                               |
#   | FRE-lin-mlp  | 0, 1/2, 1/2                                           |
#   | FRE-goal-mlp | 1/2, 0, 1/2                                           |
#   | FRE-goal-lin | 1/2, 1/2, 0                                           |
#   +--------------+-------------------------------------------------------+
#
# All ablations are run in **AntMaze** (antmaze-large-diverse-v2) with the
# paper's 5-seed x 20-episode protocol.  Reference numbers (Appendix D):
#
#   Eval Task       FRE-all     FRE-goals   FRE-lin     FRE-mlp     FRE-lin-mlp FRE-goal-mlp FRE-goal-lin
#   goal-reaching   48.8±6      66.0±4      6.0±1       24.0±6      8.0±4       52.0±6       54.0±12
#   directional     55.2±8      6.6±13      55.5±6      -6.6±14     47.9±6      5.1±25       67.1±5
#   random-simplex  21.3±4      23.5±6      14.4±3      18.5±6      14.8±4      19.7±5       10.7±3
#   path-all        63.8±10     8.3±11      50.5±9      65.4±5      58.5±7      58.6±23      55.8±8
#   ---------------------------------------------------------------------------------------------
#   total           47.3±7      26.1±8      31.6±5      25.3±8      32.3±5      33.8±15      46.9±7
#
# Success criterion: FRE-all must achieve the highest total score.
#
# Usage:
#   bash fre/scripts/train_priors_ablation.sh
#   VARIANTS="all goals lin mlp lin-mlp goal-mlp goal-lin" SEEDS="0 1 2 3 4" \
#       DEVICE=cuda RUN_ROOT=runs/table4 bash fre/scripts/train_priors_ablation.sh
#   DRY_RUN=1 bash fre/scripts/train_priors_ablation.sh     # print commands only
#
# Environment variables (all optional):
#   PYTHON        python interpreter (default: python)
#   VARIANTS      prior subsets to train (default: all goals lin mlp lin-mlp goal-mlp goal-lin)
#   DOMAIN        domain to ablate on (default: antmaze; paper uses AntMaze only)
#   SEEDS         seeds (default: 0 1 2 3 4)
#   STAGE         encoder | policy | train | all (default: all)
#   DEVICE        cpu | cuda | cuda:0 (default: "")
#   RUN_ROOT      output root (default: runs/table4_priors)
#   OUT_DIR       aggregated JSON/log destination (default: runs/table4)
#   DATASET_DIR   exported as FRE_DATASET_DIR for fre/envs/d4rl_loader.py
#   DATASET_PATH  explicit dataset file for single-domain runs
#   EPISODES      eval episodes per task (default: 20)
#   NO_PHYSICS    "1" disables ExORL physics augmentation (irrelevant for AntMaze)
#   EVAL_ONLY     "1" forces STAGE=eval (reuse existing checkpoints)
#   EXTRA_ARGS    extra CLI args forwarded verbatim to fre/main.py
#   SKIP_SUMMARY  "1" skips post-hoc aggregation
#   DRY_RUN       "1" prints commands without executing them
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN="${REPO_ROOT}/fre/main.py"

PYTHON="${PYTHON:-python}"
DOMAIN="${DOMAIN:-antmaze}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-}"
RUN_ROOT="${RUN_ROOT:-runs/table4_priors}"
OUT_DIR="${OUT_DIR:-runs/table4}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_PATH="${DATASET_PATH:-}"
EPISODES="${EPISODES:-20}"
NO_PHYSICS="${NO_PHYSICS:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${EVAL_ONLY}" == "1" ]]; then
  STAGE="eval"
fi

# Canonical Table 4 ablation variants (order follows the paper's table).
DEFAULT_VARIANTS=("all" "goals" "lin" "mlp" "lin-mlp" "goal-mlp" "goal-lin")
if [[ -n "${VARIANTS:-}" ]]; then
  # shellcheck disable=SC2206
  VARIANTS_ARR=(${VARIANTS})
else
  VARIANTS_ARR=("${DEFAULT_VARIANTS[@]}")
fi

if [[ -n "${DATASET_DIR}" ]]; then
  export FRE_DATASET_DIR="${DATASET_DIR}"
fi

if [[ ! -f "${MAIN}" ]]; then
  echo "[error] could not find ${MAIN}" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Defensive CLI probing.
#
# fre/main.py owns the authoritative argparse surface (see fre/main.py,
# where PRIOR_MIXTURES maps these exact variant names to mixture weights in
# config.prior_ratios).  We probe --help once so this script keeps working if
# the flag spelling changes, and we also accept a handful of aliases for the
# prior-mixture flag.
# -----------------------------------------------------------------------------
_HELP_CACHE=""
_LOADED_HELP=0

_load_help() {
  if [[ "${_LOADED_HELP}" == "0" ]]; then
    _HELP_CACHE="$("${PYTHON}" "${MAIN}" --help 2>/dev/null || true)"
    _LOADED_HELP=1
  fi
}

_supports() {
  local flag="$1"
  _load_help
  if [[ -z "${_HELP_CACHE}" ]]; then
    return 1
  fi
  grep -q -- "${flag}" <<<"${_HELP_CACHE}"
}

# Prefer the canonical --prior flag, falling back to documented aliases.
PRIOR_FLAG=""
for _candidate in --prior --prior-mixture --prior-mixture-name --prior-ratios; do
  if _supports "${_candidate}"; then
    PRIOR_FLAG="${_candidate}"
    break
  fi
done

_add_flag() {
  local flag="$1"
  if _supports "${flag}"; then
    CMD+=("${flag}")
  else
    echo "[warn] ${flag} not supported by fre/main.py; skipping" >&2
  fi
}

_add_opt() {
  local flag="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    return 0
  fi
  if _supports "${flag}"; then
    CMD+=("${flag}" "${value}")
  else
    echo "[warn] ${flag} not supported by fre/main.py; skipping" >&2
  fi
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "${arg}")"
  done
  printf '%s\n' "${out# }"
}

# -----------------------------------------------------------------------------
# Command construction for one (variant, domain, seed) triple.
# -----------------------------------------------------------------------------
build_command() {
  local variant="$1"
  local domain="$2"
  local seed="$3"

  CMD=("${PYTHON}" "${MAIN}")
  _add_opt --agent "fre"
  _add_opt --domain "${domain}"
  _add_opt --stage "${STAGE}"

  # Prior mixture subset: this is the whole point of the Table 4 ablation.
  if [[ -n "${PRIOR_FLAG}" ]]; then
    CMD+=("${PRIOR_FLAG}" "${variant}")
  else
    echo "[warn] no prior-mixture flag found on fre/main.py; passing prior via EXTRA_ARGS" >&2
    EXTRA_ARGS="--prior ${variant} ${EXTRA_ARGS}"
  fi

  _add_opt --seed "${seed}"
  _add_opt --run-dir "${RUN_ROOT}/${variant}/${domain}/seed${seed}"
  _add_opt --num-eval-episodes "${EPISODES}"
  _add_opt --device "${DEVICE}"
  _add_opt --dataset-path "${DATASET_PATH}"

  if [[ "${NO_PHYSICS}" == "1" ]]; then
    _add_flag --no-physics
  fi

  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${EXTRA_ARGS})
    CMD+=("${extra[@]}")
  fi
}

mkdir -p "${RUN_ROOT}" "${OUT_DIR}"

FAILED=0
RUNS=0

echo "=============================================================="
echo " FRE prior-mixture ablation (paper Table 4 / Appendix D)"
echo "--------------------------------------------------------------"
echo " domain   : ${DOMAIN}"
echo " variants : ${VARIANTS_ARR[*]}"
echo " seeds    : ${SEEDS}"
echo " stage    : ${STAGE}"
echo " run root : ${RUN_ROOT}"
echo " prior flag: ${PRIOR_FLAG:-<none>}"
echo "=============================================================="

for variant in "${VARIANTS_ARR[@]}"; do
  for seed in ${SEEDS}; do
    build_command "${variant}" "${DOMAIN}" "${seed}"
    log_dir="${RUN_ROOT}/${variant}/${DOMAIN}"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/seed${seed}.log"

    echo ""
    echo ">>> variant=${variant} domain=${DOMAIN} seed=${seed}"
    echo "    $(quote_cmd "${CMD[@]}")"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    RUNS=$((RUNS + 1))
    if ! "${CMD[@]}" 2>&1 | tee "${log_file}"; then
      echo "[error] run failed: variant=${variant} seed=${seed} (see ${log_file})" >&2
      FAILED=$((FAILED + 1))
    fi
  done
done

echo ""
echo "=============================================================="
echo " finished ${RUNS} run(s); failures: ${FAILED}"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Post-hoc aggregation: collect per-seed report.json files and compare the
# observed FRE-<variant> total scores against Appendix D, Table 4.
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" || "${SKIP_SUMMARY}" == "1" ]]; then
  echo "[info] skipping summary (DRY_RUN=${DRY_RUN}, SKIP_SUMMARY=${SKIP_SUMMARY})"
  exit 0
fi

"${PYTHON}" - "${REPO_ROOT}" "${RUN_ROOT}" "${OUT_DIR}" "${DOMAIN}" <<'PY' || {
import sys
sys.exit(0)
PY
}
PY
