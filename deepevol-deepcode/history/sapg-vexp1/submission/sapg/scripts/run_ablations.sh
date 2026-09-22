#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh — Reproduce SAPG ablation experiments (Figure 6)
# =============================================================================
#
# Paper: "SAPG: Split and Aggregate Policy Gradients"
# Section 6.3 / Figure 6 ablations:
#   1. Symmetric aggregation          (aggregation=symmetric)
#   2. High off-policy ratio          (subsample=false, i.e. no subsampling)
#   3. No off-policy combination      (aggregation=none)
#   4. Entropy coefficient grid       (entropy_coef in {0.0, 0.003, 0.005})
#
# Each ablation is run across all 5 tasks and 5 seeds, then aggregated into
# per-task figures and a summary JSON, mirroring scripts/run_all.sh.
#
# Usage:
#   bash scripts/run_ablations.sh
#   bash scripts/run_ablations.sh --tasks allegrokuka_reorientation --seeds 0
#   bash scripts/run_ablations.sh --ablations symmetric,no_off_policy
#   bash scripts/run_ablations.sh --num-envs 256 --dry-run
#
# Flags:
#   --tasks <comma-list>       restrict tasks (default: all 5)
#   --seeds <comma-list>       restrict seeds (default: 0,1,2,3,4)
#   --ablations <comma-list>   restrict ablations (default: all 4)
#   --num-envs <N>             override env count (smoke test / CPU fallback)
#   --output-root <dir>        results root (default: runs_ablations)
#   --log-dir <dir>            log directory (default: logs_ablations)
#   --dry-run                  print commands only
#   --skip-existing            skip runs with existing checkpoint_latest.pt
#   -h | --help                usage
#
# Environment variables:
#   SAPG_PYTHON                python interpreter (default: python)
#   SAPG_EXTRA_ARGS            extra args appended to every run
#   SAPG_FORCE_DUMMY_ENV       force NumPy CPU fallback env
# =============================================================================

set -u

# ---------------------------------------------------------------------------
# Resolve repository root (this script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${SAPG_PYTHON:-python}"
EXTRA_ARGS="${SAPG_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALL_TASKS="allegrokuka_regrasping,allegrokuka_throw,allegrokuka_reorientation,shadowhand,allegrohand"
ALL_SEEDS="0,1,2,3,4"
ALL_ABLATIONS="symmetric,high_off_policy,no_off_policy,entropy"

TASKS="${ALL_TASKS}"
SEEDS="${ALL_SEEDS}"
ABLATIONS="${ALL_ABLATIONS}"
NUM_ENVS=""
OUTPUT_ROOT="runs_ablations"
LOG_DIR="logs_ablations"
DRY_RUN=0
SKIP_EXISTING=0

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)         TASKS="$2"; shift 2 ;;
        --seeds)         SEEDS="$2"; shift 2 ;;
        --ablations)     ABLATIONS="$2"; shift 2 ;;
        --num-envs)      NUM_ENVS="$2"; shift 2 ;;
        --output-root)   OUTPUT_ROOT="$2"; shift 2 ;;
        --log-dir)       LOG_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
config_for_task() {
    # Map a task name to its YAML config path.
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

metric_for_task() {
    # Hard tasks report successes; easy tasks report episode reward.
    case "$1" in
        allegrokuka_regrasping|allegrokuka_throw|allegrokuka_reorientation)
            echo "successes" ;;
        *)
            echo "episode_reward" ;;
    esac
}

# Build the extra CLI overrides for a given ablation variant.
# Echoes a space-separated string of "--key value" pairs (may be empty).
ablation_overrides() {
    case "$1" in
        symmetric)
            echo "--aggregation symmetric" ;;
        high_off_policy)
            echo "--aggregation leader_follower --no-subsample" ;;
        no_off_policy)
            echo "--aggregation none" ;;
        entropy)
            # Entropy grid is expanded separately (see run loop).
            echo "" ;;
        *)
            echo "" ;;
    esac
}

# Entropy coefficient values for the entropy ablation.
entropy_coefs() {
    echo "0.0 0.003 0.005"
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=============================================================="
echo " SAPG ablation experiments (Figure 6)"
echo "--------------------------------------------------------------"
echo " tasks      : ${TASKS}"
echo " seeds      : ${SEEDS}"
echo " ablations  : ${ABLATIONS}"
echo " num-envs   : ${NUM_ENVS:-<config default>}"
echo " output-root: ${OUTPUT_ROOT}"
echo " log-dir    : ${LOG_DIR}"
echo " python     : ${PYTHON}"
echo " dry-run    : ${DRY_RUN}"
echo "=============================================================="

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a TASK_ARR <<< "${TASKS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
IFS=',' read -r -a ABL_ARR <<< "${ABLATIONS}"

FAILED=0
TOTAL=0

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------
for ABL in "${ABL_ARR[@]}"; do
    OVERRIDES="$(ablation_overrides "${ABL}")"

    # The entropy ablation sweeps the coefficient; all others run once.
    if [[ "${ABL}" == "entropy" ]]; then
        COEF_LIST="$(entropy_coefs)"
    else
        COEF_LIST="__none__"
    fi

    for COEF in ${COEF_LIST}; do
        if [[ "${COEF}" == "__none__" ]]; then
            VARIANT="${ABL}"
            COEF_ARGS=""
        else
            VARIANT="${ABL}_coef${COEF}"
            COEF_ARGS="--entropy-coef ${COEF}"
        fi

        for TASK in "${TASK_ARR[@]}"; do
            CONFIG="$(config_for_task "${TASK}")"

            for SEED in "${SEED_ARR[@]}"; do
                RUN_DIR="${OUTPUT_ROOT}/${VARIANT}/${TASK}_seed${SEED}"
                LOG_FILE="${LOG_DIR}/${VARIANT}_${TASK}_seed${SEED}.log"

                if [[ "${SKIP_EXISTING}" -eq 1 && -f "${RUN_DIR}/checkpoint_latest.pt" ]]; then
                    echo "[skip] ${VARIANT} ${TASK} seed=${SEED} (checkpoint exists)"
                    continue
                fi

                CMD=("${PYTHON}" main.py
                     --method sapg
                     --task "${TASK}"
                     --config "${CONFIG}"
                     --seed "${SEED}"
                     --output-dir "${RUN_DIR}")

                if [[ -n "${NUM_ENVS}" ]]; then
                    CMD+=(--num-envs "${NUM_ENVS}")
                fi
                if [[ -n "${OVERRIDES}" ]]; then
                    # shellcheck disable=SC2206
                    CMD+=(${OVERRIDES})
                fi
                if [[ -n "${COEF_ARGS}" ]]; then
                    # shellcheck disable=SC2206
                    CMD+=(${COEF_ARGS})
                fi
                if [[ -n "${EXTRA_ARGS}" ]]; then
                    # shellcheck disable=SC2206
                    CMD+=(${EXTRA_ARGS})
                fi

                TOTAL=$((TOTAL + 1))
                echo "--------------------------------------------------------------"
                echo "[run ${TOTAL}] ${VARIANT} | ${TASK} | seed=${SEED}"
                echo "  ${CMD[*]}"

                if [[ "${DRY_RUN}" -eq 1 ]]; then
                    continue
                fi

                mkdir -p "${RUN_DIR}"
                if ! "${CMD[@]}" > "${LOG_FILE}" 2>&1; then
                    echo "  !! FAILED (see ${LOG_FILE})"
                    FAILED=$((FAILED + 1))
                else
                    echo "  ok -> ${RUN_DIR}"
                fi
            done
        done
    done
done

echo "=============================================================="
echo " Finished: ${TOTAL} run(s), ${FAILED} failure(s)"
echo "=============================================================="

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run: skipping aggregation)"
    exit 0
fi

if [[ "${FAILED}" -ne 0 ]]; then
    echo "Skipping aggregation because ${FAILED} run(s) failed." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Aggregation: build per-task figures + summary JSON
# ---------------------------------------------------------------------------
echo "Aggregating results into figures/ ..."

"${PYTHON}" - "${OUTPUT_ROOT}" "${TASKS}" "${ABLATIONS}" <<'PY'
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from utils.logger import aggregate_runs, plot_curves  # noqa: E402

output_root = sys.argv[1]
tasks = [t for t in sys.argv[2].split(",") if t]
ablations = [a for a in sys.argv[3].split(",") if a]

HARD = {
    "allegrokuka_regrasping",
    "allegrokuka_throw",
    "allegrokuka_reorientation",
}


def metric_for(task):
    return "successes" if task in HARD else "episode_reward"


def load_metric(path, metric):
    """Load a metric curve from a run's metrics.json."""
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in (metric, "eval/" + metric, "train/" + metric):
        if key in data:
            series = data[key]
            if isinstance(series, dict):
                series = series.get("values", series.get("y"))
            if isinstance(series, list) and series:
                return [float(v) for v in series]
    return None


os.makedirs("figures", exist_ok=True)
summary = {}

for task in tasks:
    metric = metric_for(task)
    curves = {}

    for abl in ablations:
        # Collect all variants belonging to this ablation (e.g. entropy_coef*).
        pattern = os.path.join(output_root, abl + "*", task + "_seed*", "metrics.json")
        runs = []
        for path in sorted(glob.glob(pattern)):
            series = load_metric(path, metric)
            if series:
                runs.append(series)
        if runs:
            curves[abl] = runs

    if not curves:
        print("  [warn] no data for task %s" % task)
        continue

    # Aggregate each ablation across seeds onto a common grid.
    agg = {}
    for name, runs in curves.items():
        try:
            x, mean, stderr = aggregate_runs(runs, num_points=200)
            agg[name] = {"x": x.tolist(), "mean": mean.tolist(), "stderr": stderr.tolist()}
        except Exception as exc:  # pragma: no cover - defensive
            print("  [warn] aggregation failed for %s/%s: %s" % (task, name, exc))

    if not agg:
        continue

    # Plot (Figure 6 style).
    plot_curves(
        {name: [d["mean"] for d in [v]] for name, v in agg.items()},
        xlabel="transitions",
        ylabel=metric,
        title="SAPG ablations - %s" % task,
        filename=os.path.join("figures", "%s_ablations.png" % task),
    )

    summary[task] = {
        "metric": metric,
        "ablations": {
            name: {
                "final_mean": float(v["mean"][-1]) if v["mean"] else None,
                "final_stderr": float(v["stderr"][-1]) if v["stderr"] else None,
                "num_seeds": len(curves[name]),
            }
            for name, v in agg.items()
        },
    }

with open(os.path.join("figures", "ablations_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("Wrote figures/ablations_summary.json")
PY

echo "Done. Figures in figures/, summary in figures/ablations_summary.json"
