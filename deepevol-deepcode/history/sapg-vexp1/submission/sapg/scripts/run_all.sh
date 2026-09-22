#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- Reproduce the main SAPG results (Table 1, Figure 5).
#
# Runs the full experiment matrix:
#     5 tasks  x  4 methods  x  5 seeds  =  100 runs
#
# Tasks (canonical names):
#     allegrokuka_regrasping      (hard, metric = successes)
#     allegrokuka_throw           (hard, metric = successes)
#     allegrokuka_reorientation   (hard, metric = successes)
#     shadowhand                  (easy, metric = episode reward)
#     allegrohand                 (easy, metric = episode reward)
#
# Methods:
#     sapg    -- Split and Aggregate Policy Gradients (ours)
#     ppo     -- vanilla PPO with batch scaled to 24576 envs
#     dexpbt  -- population-based training (Petrenko et al., 2023)
#     pql     -- parallel Q-learning (Li et al., 2023)
#
# Usage:
#     bash scripts/run_all.sh                 # full matrix
#     bash scripts/run_all.sh --dry-run       # print commands only
#     bash scripts/run_all.sh --methods sapg  # restrict methods
#     bash scripts/run_all.sh --tasks allegrohand --seeds 0
#     bash scripts/run_all.sh --num-envs 256  # smoke test / CPU fallback
#
# Environment variables:
#     SAPG_FORCE_DUMMY_ENV=1   force the NumPy CPU fallback env (no IsaacGym)
#     SAPG_PYTHON=python3      interpreter to use (default: python)
#     SAPG_EXTRA_ARGS="..."    extra CLI args appended to every run
# =============================================================================

set -u  # note: deliberately NOT `set -e` -- we want to continue past failures

# ---------------------------------------------------------------------------
# Locate repository root (this script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON="${SAPG_PYTHON:-python}"
EXTRA_ARGS="${SAPG_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASKS=(
  "allegrokuka_regrasping"
  "allegrokuka_throw"
  "allegrokuka_reorientation"
  "shadowhand"
  "allegrohand"
)
METHODS=("sapg" "ppo" "dexpbt" "pql")
SEEDS=(0 1 2 3 4)

NUM_ENVS=""
OUTPUT_ROOT="runs"
DRY_RUN=0
SKIP_EXISTING=0
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      shift
      IFS=',' read -r -a TASKS <<< "$1"
      ;;
    --methods)
      shift
      IFS=',' read -r -a METHODS <<< "$1"
      ;;
    --seeds)
      shift
      IFS=',' read -r -a SEEDS <<< "$1"
      ;;
    --num-envs)
      shift
      NUM_ENVS="$1"
      ;;
    --output-root)
      shift
      OUTPUT_ROOT="$1"
      ;;
    --log-dir)
      shift
      LOG_DIR="$1"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
  shift
done

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Per-task config file mapping
# ---------------------------------------------------------------------------
config_for_task() {
  case "$1" in
    allegrokuka_regrasping|allegrokuka_throw|allegrokuka_reorientation)
      echo "configs/allegrokuka.yaml" ;;
    shadowhand)
      echo "configs/shadowhand.yaml" ;;
    allegrohand)
      echo "configs/allegrohand.yaml" ;;
    *)
      echo "configs/sapg.yaml" ;;
  esac
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
TOTAL=$(( ${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
COUNT=0
FAILED=0
START_TIME=$(date +%s)

echo "======================================================================"
echo " SAPG full reproduction matrix"
echo "   tasks   : ${TASKS[*]}"
echo "   methods : ${METHODS[*]}"
echo "   seeds   : ${SEEDS[*]}"
echo "   runs    : ${TOTAL}"
echo "   python  : ${PYTHON}"
echo "   root    : ${ROOT_DIR}"
[[ -n "${NUM_ENVS}" ]] && echo "   num-envs: ${NUM_ENVS} (override)"
[[ "${DRY_RUN}" -eq 1 ]] && echo "   MODE    : DRY RUN"
echo "======================================================================"

for task in "${TASKS[@]}"; do
  cfg="$(config_for_task "${task}")"
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      run_name="${method}_${task}_seed${seed}"
      out_dir="${OUTPUT_ROOT}/${run_name}"
      log_file="${LOG_DIR}/${run_name}.log"

      cmd=(
        "${PYTHON}" main.py
        --method "${method}"
        --task "${task}"
        --config "${cfg}"
        --seed "${seed}"
        --output-dir "${out_dir}"
      )
      [[ -n "${NUM_ENVS}" ]] && cmd+=(--num-envs "${NUM_ENVS}")
      # shellcheck disable=SC2206
      [[ -n "${EXTRA_ARGS}" ]] && cmd+=(${EXTRA_ARGS})

      printf '[%3d/%3d] %s\n' "${COUNT}" "${TOTAL}" "${run_name}"

      if [[ "${SKIP_EXISTING}" -eq 1 && -f "${out_dir}/checkpoint_latest.pt" ]]; then
        echo "          -> skipping (checkpoint exists)"
        continue
      fi

      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "          ${cmd[*]}"
        continue
      fi

      echo "          log: ${log_file}"
      if "${cmd[@]}" > "${log_file}" 2>&1; then
        echo "          -> OK"
      else
        rc=$?
        echo "          -> FAILED (exit ${rc}); see ${log_file}" >&2
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================================================"
echo " Finished: ${COUNT} runs, ${FAILED} failed, ${ELAPSED}s elapsed"
echo " Results : ${OUTPUT_ROOT}/"
echo " Logs    : ${LOG_DIR}/"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Post-processing: aggregate multi-seed curves for Figure 5 / Table 1
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 0 && "${FAILED}" -eq 0 ]]; then
  echo
  echo "Aggregating results across seeds (Table 1 / Figure 5)..."
  "${PYTHON}" - <<'PYEOF' || echo "  (aggregation skipped: see traceback above)"
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

try:
    from utils.logger import aggregate_runs, plot_curves
except Exception as exc:  # pragma: no cover
    print(f"  could not import utils.logger: {exc}")
    raise SystemExit(0)

HARD = ("allegrokuka_regrasping", "allegrokuka_throw", "allegrokuka_reorientation")
EASY = ("shadowhand", "allegrohand")
METHODS = ("sapg", "ppo", "dexpbt", "pql")

summary = {}
for task in HARD + EASY:
    metric = "successes" if task in HARD else "episode_reward"
    curves = {}
    for method in METHODS:
        runs = []
        for path in sorted(glob.glob(f"runs/{method}_{task}_seed*/metrics.json")):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception:
                continue
            series = data.get(metric) or data.get(f"eval/{metric}") or []
            if isinstance(series, list) and series:
                runs.append([float(v) for v in series])
        if runs:
            curves[method] = runs
    if not curves:
        continue

    os.makedirs("figures", exist_ok=True)
    plot_curves(
        curves,
        xlabel="transitions",
        ylabel=metric,
        title=task,
        filename=f"figures/{task}_{metric}.png",
    )

    summary[task] = {}
    for method, runs in curves.items():
        _, mean, stderr = aggregate_runs(runs, num_points=200)
        summary[task][method] = {
            "final_mean": float(mean[-1]),
            "final_stderr": float(stderr[-1]),
            "num_seeds": len(runs),
        }
        print(
            f"  {task:28s} {method:7s} "
            f"{mean[-1]:12.4g} +/- {stderr[-1]:.3g}  (n={len(runs)})"
        )

with open("figures/summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("Wrote figures/summary.json")
PYEOF
fi

exit 0
