#!/usr/bin/env bash
# =============================================================================
# run_baselines.sh -- Train and evaluate the Table 1 comparison methods
# =============================================================================
#
# Reproduces the baseline half of Table 1 of
#   "Zero-Shot Reinforcement Learning via Functional Reward Encodings" (FRE),
#   ICML 2024, https://github.com/kvfrans/fre
#
# Methods (see FRE Sec. 5.2 and the addendum):
#   gc_iql : Goal-Conditioned IQL (Kostrikov et al., 2021) re-implemented in the
#            FRE codebase. Goal concatenated to the observation; reward is 0 if
#            s == goal else -1; hindsight goal sampling uses
#            p_current = 0.2, p_geometric = 0.5, p_random = 0.3.
#   gc_bc  : Goal-Conditioned Behavioral Cloning -- MLP [512,512,512] with ReLU
#            and LayerNorm pre-activation, Gaussian policy, log-std clamped at
#            -5.0, loss = -E[log pi(a | s, g)], geometric future-goal sampling
#            only.
#   opal   : OPAL (Ajay et al., 2020) re-implemented with the SAME transformer
#            encoder architecture as FRE. Because OPAL does not solve the
#            zero-shot problem, it is compared under a *privileged* protocol:
#            OPAL-10 draws 10 skills from N(0, I) and keeps the best rollout.
#   fb     : Forward-Backward (Touati & Ollivier, 2021) via the code released by
#            (Touati et al., 2022), https://github.com/facebookresearch/controllable_agent
#   sf     : Successor Features (Barreto et al., 2017; Borsa et al., 2018) from
#            the same codebase, with ICM features (Pathak et al., 2017), which
#            is reported to be the strongest variant on ExORL Walker/Cheetah.
#
# Evaluation protocol (paper Sec. 5 + 5.2, and the addendum):
#   * 20 evaluation episodes per task, mean reported;
#   * 5 random training seeds, standard deviation across seeds reported;
#   * normalised returns on a 0-100 scale;
#   * FRE / GC-IQL / GC-BC receive 32 reward-annotated samples at test time,
#     while FB / SF are given 5120 reward samples (linear-regression test-time
#     adaptation).  This script forwards the matching `--num-eval-samples`.
#
# Usage
# -----
#   bash fre/scripts/run_baselines.sh                      # every baseline, all domains
#   AGENTS="fb sf" bash fre/scripts/run_baselines.sh       # only FB and SF
#   AGENT=gc_iql DOMAINS=kitchen SEEDS="0 1" \
#       bash fre/scripts/run_baselines.sh                  # single method/domain sweep
#   DRY_RUN=1 bash fre/scripts/run_baselines.sh            # print commands only
#
# Environment variables (all optional)
# ------------------------------------
#   PYTHON         python interpreter                    (default: python)
#   AGENTS         space-separated baselines to run      (default: gc_iql gc_bc opal fb sf)
#   AGENT          single baseline; overrides AGENTS
#   DOMAINS        space-separated domains               (default: antmaze exorl_walker
#                                                         exorl_cheetah kitchen)
#   SEEDS          space-separated seeds                 (default: 0 1 2 3 4)
#   STAGE          encoder | policy | train | all | eval (default: all)
#   DEVICE         cpu | cuda | cuda:0                   (default: "" -> main.py default)
#   RUN_ROOT       output root for checkpoints/reports   (default: runs/baselines)
#   OUT_DIR        directory for aggregated JSON tables  (default: runs/table1)
#   DATASET_DIR    exported as FRE_DATASET_DIR           (default: "")
#   DATASET_PATH   explicit dataset file for one domain  (default: "")
#   EPISODES       evaluation episodes per task          (default: 20)
#   FRE_SAMPLES    reward samples for FRE-like agents    (default: 32)
#   FB_SF_SAMPLES  reward samples for FB/SF              (default: 5120)
#   OPAL_SKILLS    privileged skill count for OPAL       (default: 10)
#   CA_DIR         checkout of controllable_agent        (default: third_party/controllable_agent)
#   NO_PHYSICS     1 disables ExORL physics augmentation (default: 0)
#   EVAL_ONLY      1 -> STAGE=eval (reuse checkpoints)   (default: 0)
#   SKIP_SUMMARY   1 skips the post-hoc aggregation      (default: 0)
#   EXTRA_ARGS     extra CLI args forwarded verbatim
#   DRY_RUN        1 prints commands without running them (default: 0)
#
# Outputs
# -------
#   ${RUN_ROOT}/<agent>/<domain>/seed<seed>/report.json   per-run results
#   ${OUT_DIR}/<agent>_table1.json                        aggregated scores
#   ${OUT_DIR}/<agent>_<domain>_seed<seed>.log            per-run logs
#   ${OUT_DIR}/baselines_summary.json                     all agents, one file
# =============================================================================

set -euo pipefail

PYTHON=${PYTHON:-python}
AGENTS=${AGENTS:-"gc_iql gc_bc opal fb sf"}
AGENT=${AGENT:-}
DOMAINS=${DOMAINS:-"antmaze exorl_walker exorl_cheetah kitchen"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-}
RUN_ROOT=${RUN_ROOT:-runs/baselines}
OUT_DIR=${OUT_DIR:-runs/table1}
DATASET_DIR=${DATASET_DIR:-}
DATASET_PATH=${DATASET_PATH:-}
EPISODES=${EPISODES:-20}
FRE_SAMPLES=${FRE_SAMPLES:-32}
FB_SF_SAMPLES=${FB_SF_SAMPLES:-5120}
OPAL_SKILLS=${OPAL_SKILLS:-10}
CA_DIR=${CA_DIR:-third_party/controllable_agent}
NO_PHYSICS=${NO_PHYSICS:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
SKIP_SUMMARY=${SKIP_SUMMARY:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
DRY_RUN=${DRY_RUN:-0}

# `AGENT` (singular) overrides the sweep list when provided.
if [[ -n "${AGENT}" ]]; then
  AGENTS="${AGENT}"
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  STAGE=eval
fi

# Repo root = two levels up from this script ( <root>/fre/scripts/run_baselines.sh ).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN="${REPO_ROOT}/fre/main.py"

if [[ ! -f "${MAIN}" ]]; then
  echo "[error] cannot find ${MAIN}" >&2
  exit 2
fi

# Propagate the dataset root to fre/envs/d4rl_loader.py (FRE_DATASET_DIR contract).
if [[ -n "${DATASET_DIR}" ]]; then
  export FRE_DATASET_DIR="${DATASET_DIR}"
fi
if [[ -n "${CA_DIR}" ]]; then
  export CONTROLLABLE_AGENT_DIR="${CA_DIR}"
fi

mkdir -p "${OUT_DIR}"

# -----------------------------------------------------------------------------
# Defensive CLI probing
# -----------------------------------------------------------------------------
# `fre/main.py` is a single argparse entry point shared by all baselines.  Its
# flag surface may grow (or shrink) between revisions, so we cache `--help`
# once and only forward flags it actually advertises.  A missing flag produces a
# warning rather than aborting the whole sweep.
_HELP_CACHE=""
_load_help() {
  if [[ -z "${_HELP_CACHE}" ]]; then
    _HELP_CACHE="$("${PYTHON}" "${MAIN}" --help 2>&1 || true)"
  fi
}

_supports() {
  _load_help
  grep -q -- "$1" <<<"${_HELP_CACHE}"
}

CMD=()
_add_flag() {
  if _supports "$1"; then
    CMD+=("$1")
  else
    echo "[warn] ${MAIN} does not accept $1; skipping" >&2
  fi
}

_add_opt() {
  if _supports "$1"; then
    CMD+=("$1" "$2")
  else
    echo "[warn] ${MAIN} does not accept $1; skipping ($2)" >&2
  fi
}

quote_cmd() {
  local out="" a
  for a in "$@"; do
    printf -v a '%q' "$a"
    out+=" ${a}"
  done
  printf '%s' "${out# }"
}

# -----------------------------------------------------------------------------
# Per-baseline evaluation sample budget
# -----------------------------------------------------------------------------
# Table 1 caption / Sec. 5.2: FB and SF perform linear-regression test-time
# adaptation and are given 5120 reward samples; FRE-family methods get 32.
_agent_eval_samples() {
  case "$1" in
    fb|forward_backward|forward-backward|sf|successor_features|successor-features)
      echo "${FB_SF_SAMPLES}" ;;
    *)
      echo "${FRE_SAMPLES}" ;;
  esac
}

# -----------------------------------------------------------------------------
# Command construction
# -----------------------------------------------------------------------------
build_command() {
  local agent="$1" domain="$2" seed="$3"
  local run_dir="${RUN_ROOT}/${agent}/${domain}/seed${seed}"
  local samples
  samples="$(_agent_eval_samples "${agent}")"

  CMD=("${PYTHON}" "${MAIN}")
  _add_flag --no-eval-placeholder 2>/dev/null || true   # no-op probe keeps cache warm
  _add_opt --agent "${agent}"
  _add_opt --domain "${domain}"
  _add_opt --stage "${STAGE}"
  _add_opt --seed "${seed}"
  _add_opt --run-dir "${run_dir}"
  _add_opt --num-eval-episodes "${EPISODES}"
  _add_opt --num-eval-samples "${samples}"
  [[ -n "${DEVICE}" ]] && _add_opt --device "${DEVICE}"
  [[ -n "${DATASET_PATH}" ]] && _add_opt --dataset-path "${DATASET_PATH}"

  # OPAL's privileged protocol: 10 skills sampled from N(0, I), best rollout
  # kept (this is the "OPAL-10" column of Table 1).
  if [[ "${agent}" == "opal" ]]; then
    _add_opt --num-skills "${OPAL_SKILLS}"
  fi

  # ExORL encoder augmentation (walker: horizontal_velocity, torso_upright,
  # torso_height; cheetah: speed).  Appendix C.2.
  if [[ "${NO_PHYSICS}" == "1" ]]; then
    _add_flag --no-physics
  fi

  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(${EXTRA_ARGS})
  fi

  # Drop the harmless placeholder probe if it ever made it through.
  local cleaned=() tok
  for tok in "${CMD[@]}"; do
    [[ "${tok}" == "--no-eval-placeholder" ]] && continue
    cleaned+=("${tok}")
  done
  CMD=("${cleaned[@]}")
}

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
echo "=============================================================="
echo " FRE Table 1 -- baseline comparison sweep"
echo "=============================================================="
echo " repo root     : ${REPO_ROOT}"
echo " agents        : ${AGENTS}"
echo " domains       : ${DOMAINS}"
echo " seeds         : ${SEEDS}"
echo " stage         : ${STAGE}"
echo " episodes      : ${EPISODES}"
echo " FRE samples   : ${FRE_SAMPLES}"
echo " FB/SF samples : ${FB_SF_SAMPLES}"
echo " OPAL skills   : ${OPAL_SKILLS}"
echo " device        : ${DEVICE:-<default>}"
echo " run root      : ${RUN_ROOT}"
echo " out dir       : ${OUT_DIR}"
echo " dataset dir   : ${DATASET_DIR:-<default>}"
echo " controllable  : ${CA_DIR}"
echo " dry run       : ${DRY_RUN}"
echo "=============================================================="

TOTAL=0
FAILED=0

for agent in ${AGENTS}; do
  for domain in ${DOMAINS}; do
    for seed in ${SEEDS}; do
      TOTAL=$((TOTAL + 1))
      build_command "${agent}" "${domain}" "${seed}"
      log_file="${OUT_DIR}/${agent}_${domain}_seed${seed}.log"
      echo "[run] $(quote_cmd "${CMD[@]}")"

      if [[ "${DRY_RUN}" == "1" ]]; then
        continue
      fi

      mkdir -p "$(dirname "${log_file}")"
      if ! "${CMD[@]}" >"${log_file}" 2>&1; then
        echo "[fail] ${agent}/${domain}/seed${seed} -- see ${log_file}" >&2
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo "=============================================================="
echo "[done] ${TOTAL} run(s) issued, ${FAILED} failure(s)"
echo "=============================================================="

if [[ "${DRY_RUN}" == "1" || "${SKIP_SUMMARY}" == "1" ]]; then
  exit 0
fi

# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
# Collects every `report.json` produced under RUN_ROOT, groups per agent, and
# prints observed scores next to the published Table 1 numbers.
"${PYTHON}" - "${REPO_ROOT}" "${RUN_ROOT}" "${OUT_DIR}" "${AGENTS}" <<'PY'
import json
import os
import sys

repo_root, run_root, out_dir, agents = sys.argv[1:5]
agents = agents.split()

sys.path.insert(0, repo_root)

try:
    from fre.utils.normalization import (  # noqa: E402
        TABLE1_AGGREGATE_TARGETS,
        TABLE1_FRE_TARGETS,
        aggregate_rows,
        format_mean_std,
    )
    HAVE_TARGETS = True
except Exception as err:  # pragma: no cover - reporting must never hard-fail
    print(f"[warn] could not import reference targets: {err}")
    HAVE_TARGETS = False

    def format_mean_std(mean, std, digits=1):
        if std is None:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def find_reports(root):
    """Return {agent: {seed: payload}} by scanning RUN_ROOT for report.json."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if "report.json" not in filenames:
            continue
        path = os.path.join(dirpath, "report.json")
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        agent = payload.get("agent")
        seed = payload.get("seed")
        parts = os.path.normpath(dirpath).split(os.sep)
        if agent is None and len(parts) >= 3:
            agent = parts[-3]
        if seed is None and len(parts) >= 1 and parts[-1].startswith("seed"):
            try:
                seed = int(parts[-1][4:])
            except ValueError:
                seed = 0
        rows = payload.get("rows")
        if rows is None:
            rows = payload.get("results")
        if not isinstance(rows, dict):
            continue
        found.setdefault(str(agent), {})[int(seed or 0)] = rows
    return found


def reference_row(agent, row):
    if not HAVE_TARGETS:
        return None
    table = TABLE1_FRE_TARGETS
    if agent in ("fb", "forward_backward"):
        key = f"{row}"
        entry = table.get(key)
        return entry
    entry = table.get(row)
    return entry


reports = find_reports(run_root)
if not reports:
    print(f"[warn] no report.json files found under {run_root}; nothing to aggregate")
    sys.exit(0)

summary = {}
for agent in agents:
    per_seed = reports.get(agent)
    if not per_seed:
        print(f"[info] {agent}: no reports found -- skipped")
        continue

    # Build the per-seed {task -> score} mapping expected by aggregate_rows.
    seeded_results = {}
    for seed, rows in per_seed.items():
        seeded_results[seed] = {
            name: (val if isinstance(val, dict) else {"score": val})
            for name, val in rows.items()
        }

    try:
        aggregated = aggregate_rows(seeded_results)
    except Exception as err:
        print(f"[warn] {agent}: aggregation failed ({err})")
        continue

    summary[agent] = aggregated

    print()
    print(f"--- {agent} (seeds: {sorted(per_seed)}) " + "-" * 30)
    print(f"{'row':<28}{'observed':>18}{'published (FRE table)':>26}")
    for row in sorted(aggregated):
        stats = aggregated[row]
        mean = stats.get("mean")
        std = stats.get("std")
        observed = format_mean_std(mean, std) if mean is not None else "n/a"
        ref = reference_row(agent, row) if HAVE_TARGETS else None
        if ref is None:
            ref_txt = "-"
        elif isinstance(ref, (tuple, list)) and len(ref) == 2:
            ref_txt = format_mean_std(ref[0], ref[1])
        else:
            ref_txt = str(ref)
        print(f"{row:<28}{observed:>18}{ref_txt:>26}")

if not summary:
    sys.exit(0)

os.makedirs(out_dir, exist_ok=True)
summary_path = os.path.join(out_dir, "baselines_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

for agent, aggregated in summary.items():
    agent_path = os.path.join(out_dir, f"{agent}_table1.json")
    with open(agent_path, "w") as fh:
        json.dump(aggregated, fh, indent=2, default=str)

print()
print(f"[saved] {summary_path}")
PY

echo "=============================================================="
echo "[done] baseline sweep complete"
echo "=============================================================="
