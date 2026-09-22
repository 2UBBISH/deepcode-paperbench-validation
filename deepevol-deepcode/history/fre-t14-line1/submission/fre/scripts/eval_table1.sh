#!/usr/bin/env bash
#
# eval_table1.sh -- Reproduce Table 1 of
#   "Zero-Shot Reinforcement Learning via Functional Reward Encodings" (FRE, ICML 2024).
#
# Table 1 (paper Section 5.2) compares FRE against zero-shot / privileged baselines on
# three evaluation domains, normalizing all results between 0 and 100 and reporting
# mean +- std over 5 seeds (20 rollouts each, averaged):
#
#   Eval Task               FRE            FB             SF           GC-IQL   GC-BC   OPAL-10
#   ant-goal-reaching        48.8 +- 6      0.0 +- 0        0.4 +- 2     40.0 +-14 12.0+-18  19.4 +-12
#   ant-directional          55.2 +- 8      4.8 +-14        6.5 +-16     -        -        39.4 +-13
#   ant-random-simplex       21.3 +- 4      9.7 +- 2        8.5 +-10     -        -        27.3 +- 8
#   ant-path-loop            67.2 +-36     46.6 +-40       13.6 +-16     -        -        44.4 +-22
#   ant-path-edges           60.0 +-17     23.5 +-25        2.2 +- 5      -        -        85.0 +-10
#   ant-path-center          64.4 +-38     70.3 +-37       39.4 +-27     -        -        58.1 +-36
#   antmaze-all              52.8 +-18.2   25.8 +-19.8    11.8 +-12.6    -        -        45.6 +-17.0
#   exorl-walker-goals       94   +- 2     58   +-30      100   +- 0      92 +- 4  52+-18    88 +- 8
#   exorl-cheetah-goals      58   +- 8      1   +- 2        0   +- 0     100 +- 0  14+- 6     0 +- 0
#   exorl-walker-velocity    34   +-13     64   +- 1       38   +- 4      -        -         8 +- 0
#   exorl-cheetah-velocity   20   +- 2     51   +- 3       25   +- 3      -        -        17 +- 8
#   exorl-all                51.5 +- 6.3   43.4 +- 9.1    40.9 +- 1.9    -        -        28.2 +- 4.0
#   kitchen                  66   +- 3      3   +- 6        1   +- 1      59 +- 4  35+- 9   26 +-16
#   all                      57   +- 9     24   +-12       18   +- 5       -        -        33 +-12
#
# Note (Table 1 caption): "FRE utilizes only 32 examples of (state, reward) pairs during
# evaluation, while the FB and SF methods require 5120 examples to be consistent with
# prior work."  The `--num-eval-samples` knob below is set accordingly per agent.
#
# ---------------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------------
#   bash fre/scripts/eval_table1.sh                       # FRE, all domains, seeds 0-4
#   AGENT=opal bash fre/scripts/eval_table1.sh            # OPAL-10 privileged evaluation
#   AGENTS="fre gc_iql gc_bc opal" bash fre/scripts/eval_table1.sh
#   DOMAINS="antmaze" SEEDS="0" bash fre/scripts/eval_table1.sh      # quick smoke test
#   EVAL_ONLY=1 bash fre/scripts/eval_table1.sh           # reuse prior checkpoints
#   DRY_RUN=1 bash fre/scripts/eval_table1.sh             # print commands only
#
# ---------------------------------------------------------------------------------
# Environment variables (input contract)
# ---------------------------------------------------------------------------------
#   PYTHON        Python interpreter                                  (default: python)
#   AGENTS        space-separated agent list                          (default: fre)
#   AGENT         single-agent shortcut (overrides AGENTS)            (default: unset)
#   DOMAINS       space-separated domains                             (default: antmaze exorl_walker exorl_cheetah kitchen)
#   SEEDS         space-separated seeds                               (default: 0 1 2 3 4)
#   STAGE         fre/main.py stage                                    (default: all, or eval when EVAL_ONLY=1)
#   DEVICE        cpu | cuda | cuda:0                                 (default: empty -> main.py default)
#   RUN_ROOT      output directory root                               (default: runs)
#   OUT_DIR       where the aggregated table is written               (default: ${RUN_ROOT}/table1)
#   DATASET_DIR   exported as FRE_DATASET_DIR                         (default: unset)
#   DATASET_PATH  explicit dataset path for single-domain runs        (default: unset)
#   EPISODES      evaluation episodes per task                        (default: 20)
#   NO_PHYSICS    1 disables ExORL physics augmentation               (default: 0)
#   EVAL_ONLY     1 -> pass --stage eval (does not retrain)           (default: 0)
#   FB_SF_SAMPLES reward samples for FB/SF at eval (paper: 5120)      (default: 5120)
#   FRE_SAMPLES   reward samples for FRE/GC at eval (paper: 32)       (default: 32)
#   EXTRA_ARGS    extra CLI args forwarded verbatim to fre/main.py    (default: unset)
#   DRY_RUN       1 -> print commands only                            (default: 0)
#   SKIP_SUMMARY  1 -> do not render the aggregated table             (default: 0)
# ---------------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"

if [[ -n "${AGENT:-}" ]]; then
  AGENTS="${AGENT}"
else
  AGENTS="${AGENTS:-fre}"
fi
DOMAINS="${DOMAINS:-antmaze exorl_walker exorl_cheetah kitchen}"
SEEDS="${SEEDS:-0 1 2 3 4}"

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  STAGE="${STAGE:-eval}"
else
  STAGE="${STAGE:-all}"
fi

DEVICE="${DEVICE:-}"
RUN_ROOT="${RUN_ROOT:-runs}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/table1}"
EPISODES="${EPISODES:-20}"
NO_PHYSICS="${NO_PHYSICS:-0}"
FB_SF_SAMPLES="${FB_SF_SAMPLES:-5120}"
FRE_SAMPLES="${FRE_SAMPLES:-32}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

if [[ -n "${DATASET_DIR:-}" ]]; then
  export FRE_DATASET_DIR="${DATASET_DIR}"
fi

# Resolve repository root (this script lives in <root>/fre/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN="${REPO_ROOT}/fre/main.py"

echo "======================================================================"
echo " FRE -- Table 1 reproduction (Offline zero-shot RL comparisons)"
echo "======================================================================"
echo " python      : ${PYTHON}"
echo " main        : ${MAIN}"
echo " agents      : ${AGENTS}"
echo " domains     : ${DOMAINS}"
echo " seeds       : ${SEEDS}"
echo " stage       : ${STAGE}"
echo " episodes    : ${EPISODES}"
echo " eval samples: FRE/GC=${FRE_SAMPLES}  FB/SF=${FB_SF_SAMPLES}"
echo " run root    : ${RUN_ROOT}"
echo " out dir     : ${OUT_DIR}"
echo " device      : ${DEVICE:-<default>}"
echo " eval only   : ${EVAL_ONLY:-0}"
echo " dry run     : ${DRY_RUN}"
echo "======================================================================"

if [[ ! -f "${MAIN}" ]]; then
  echo "ERROR: could not find ${MAIN}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Defensive CLI probing
#
# fre/main.py is the single argparse entry point for the whole reproduction.  The
# helpers below probe its --help output once and only forward flags that actually
# exist, so this script keeps working if the argparse surface evolves.
# ---------------------------------------------------------------------------
_HELP_CACHE=""
_load_help() {
  if [[ -z "${_HELP_CACHE}" ]]; then
    _HELP_CACHE="$("${PYTHON}" "${MAIN}" --help 2>&1 || true)"
  fi
}

_supports() {
  # _supports --some-flag  -> 0 if fre/main.py accepts it
  _load_help
  if grep -q -- "$1" <<<"${_HELP_CACHE}"; then
    return 0
  fi
  return 1
}

CMD=()
_add_flag() {
  if _supports "$1"; then
    CMD+=("$1")
  else
    echo "  [warn] fre/main.py does not accept $1 -- skipping" >&2
  fi
}

_add_opt() {
  if _supports "$1"; then
    CMD+=("$1" "$2")
  else
    echo "  [warn] fre/main.py does not accept $1 -- skipping" >&2
  fi
}

_agent_eval_samples() {
  case "$1" in
    fb|sf|forward_backward|successor_features) echo "${FB_SF_SAMPLES}" ;;
    *) echo "${FRE_SAMPLES}" ;;
  esac
}

build_command() {
  local agent="$1" domain="$2" seed="$3"

  CMD=("${PYTHON}" "${MAIN}")
  _add_flag "--agent" 2>/dev/null || true
  # --agent takes a value, so add it explicitly via _add_opt semantics.
  CMD=("${PYTHON}" "${MAIN}")
  _add_opt "--agent" "${agent}"
  _add_opt "--domain" "${domain}"
  _add_opt "--stage" "${STAGE}"
  _add_opt "--seed" "${seed}"
  _add_opt "--num-eval-episodes" "${EPISODES}"
  _add_opt "--num-eval-samples" "$(_agent_eval_samples "${agent}")"

  if [[ -n "${DEVICE}" ]]; then
    _add_opt "--device" "${DEVICE}"
  fi
  if [[ -n "${DATASET_PATH:-}" ]]; then
    _add_opt "--dataset-path" "${DATASET_PATH}"
  fi
  if [[ "${NO_PHYSICS}" == "1" ]]; then
    _add_flag "--no-physics"
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(${EXTRA_ARGS})
  fi
}

quote_cmd() {
  local out=""
  local a
  for a in "$@"; do
    out+=" $(printf '%q' "${a}")"
  done
  echo "${out# }"
}

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

FAILED=0
RUNS=0

for agent in ${AGENTS}; do
  for domain in ${DOMAINS}; do
    for seed in ${SEEDS}; do
      build_command "${agent}" "${domain}" "${seed}"
      LOG="${OUT_DIR}/${agent}_${domain}_seed${seed}.log"
      RUNS=$((RUNS + 1))

      echo ""
      echo "----------------------------------------------------------------------"
      echo "[${RUNS}] agent=${agent}  domain=${domain}  seed=${seed}"
      echo "      -> $(quote_cmd "${CMD[@]}")"
      echo "      log: ${LOG}"
      echo "----------------------------------------------------------------------"

      if [[ "${DRY_RUN}" == "1" ]]; then
        continue
      fi

      if "${CMD[@]}" 2>&1 | tee "${LOG}"; then
        echo "      [ok] ${agent}/${domain}/seed${seed}"
      else
        echo "      [FAIL] ${agent}/${domain}/seed${seed} (see ${LOG})" >&2
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo ""
echo "======================================================================"
echo " Finished ${RUNS} run(s); failures: ${FAILED}"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" || "${SKIP_SUMMARY}" == "1" ]]; then
  exit 0
fi

# ---------------------------------------------------------------------------
# Aggregate per-seed reports and compare against the published Table 1 / Table 4
# numbers.  Uses the reproduction's own reporting utilities so the aggregation
# semantics (mean/std over 5 seeds, std = sample std ddof=1, domain "-all" rows
# averaged from per-task means) match fre/utils/normalization.py exactly.
# ---------------------------------------------------------------------------
"${PYTHON}" - "$RUN_ROOT" "$OUT_DIR" "$AGENTS" "$SEEDS" <<'PY'
import json
import os
import sys

run_root, out_dir, agents, seeds = sys.argv[1:5]
agents = agents.split()
seeds = seeds.split()
os.makedirs(out_dir, exist_ok=True)

# Make the repository importable regardless of the caller's cwd.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, _here)

try:
    from fre.utils.normalization import (  # type: ignore
        TABLE1_AGGREGATE_TARGETS,
        TABLE1_FRE_TARGETS,
        TABLE4_FRE_TARGETS,
        aggregate_rows,
        aggregate_task_scores,
        format_mean_std,
    )
except Exception as exc:  # pragma: no cover - reporting is best-effort
    print(f"[summary] could not import fre.utils.normalization: {exc}")
    print("[summary] skipping aggregated Table 1 rendering.")
    sys.exit(0)


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _collect(agent, seed):
    """Find a report.json for (agent, seed) anywhere under run_root."""
    matches = []
    for root, _dirs, files in os.walk(run_root):
        for fname in files:
            if fname != "report.json":
                continue
            path = os.path.join(root, fname)
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            info = " ".join(
                str(payload.get(k, ""))
                for k in ("agent", "domain", "seed")
            ) + " " + path
            if agent and agent not in str(payload.get("agent", "")) and agent not in path:
                continue
            if seed not in str(payload.get("seed", "")) and f"seed{seed}" not in path:
                continue
            matches.append((path, payload))
    if not matches:
        return None
    # Prefer the most recently written report.
    matches.sort(key=lambda item: os.path.getmtime(item[0]))
    return matches[-1][1]


def _extract_rows(payload):
    """Normalize the various report shapes into {task_name: {mean, std}}."""
    if payload is None:
        return {}
    for key in ("results", "rows", "scores", "table"):
        rows = payload.get(key)
        if isinstance(rows, dict):
            return rows
        if isinstance(rows, list):
            out = {}
            for item in rows:
                if isinstance(item, dict) and "name" in item:
                    out[str(item["name"])] = item
            if out:
                return out
    return {}


def _report_reference(agent):
    try:
        from fre.baselines import baseline_reference_row  # type: ignore

        return baseline_reference_row(agent) or {}
    except Exception:
        return {}


print("")
print("=" * 74)
print(" TABLE 1 -- observed vs. published (mean +- std, normalized 0-100)")
print("=" * 74)

for agent in agents:
    per_seed = {}
    for seed in seeds:
        payload = _collect(agent, seed)
        rows = _extract_rows(payload)
        if rows:
            per_seed[str(seed)] = rows

    if not per_seed:
        print(f"\n[{agent}] no report.json found under {run_root!r} -- nothing to aggregate.")
        continue

    agg = aggregate_task_scores(per_seed, ddof=1)
    overall = aggregate_rows(per_seed, ddof=1) if len(per_seed) > 1 else agg
    reference = _report_reference(agent)
    if not reference and agent == "fre":
        reference = dict(TABLE1_FRE_TARGETS)

    print(f"\n[{agent}]  ({len(per_seed)} seed(s) found: {', '.join(sorted(per_seed))})")
    print(f"  {'task':<26}{'observed':>16}{'published':>16}{'delta':>10}")

    for task in sorted(overall):
        cell = overall[task]
        mean = float(cell.get("mean", float("nan")))
        std = float(cell.get("std", 0.0))
        observed = format_mean_std(mean, std)
        ref = reference.get(task)
        if ref is None:
            ref_txt, delta_txt = "-", "-"
        else:
            ref_mean, ref_std = (ref if isinstance(ref, (tuple, list)) else (ref, 0.0))
            ref_txt = format_mean_std(ref_mean, ref_std)
            delta_txt = f"{mean - float(ref_mean):+.1f}"
        print(f"  {task:<26}{observed:>16}{ref_txt:>16}{delta_txt:>10}")

    if agent == "fre":
        extra = set(TABLE4_FRE_TARGETS) - set(overall)
        if extra:
            print(f"  (Table 4 prior-ablation rows also available: {sorted(extra)[:4]} ...)")

    try:
        with open(os.path.join(out_dir, f"{agent}_table1.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"agent": agent, "per_seed": per_seed, "aggregate": overall}, fh, indent=2
            )
    except Exception as exc:
        print(f"  [warn] could not write aggregate JSON: {exc}")

print("")
print("=" * 74)
print(" Published aggregates for reference:")
for row, (mean, std) in sorted(dict(TABLE1_AGGREGATE_TARGETS).items()):
    print(f"   {row:<16}{format_mean_std(mean, std)}")
print("=" * 74)
PY

exit 0
