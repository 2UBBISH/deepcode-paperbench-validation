#!/usr/bin/env bash
# =============================================================================
# RICE: A Refining scheme for ReInForcement learning with Explanation
# (Proc. 41st ICML, PMLR 235, 2024) -- full experiment orchestrator
#
# Implements the experiment plan of the paper:
#   §4.1 Experiment Setup          -> environments, baselines, metrics (fidelity)
#   §4.2 Experiment Design         -> Experiment I .. Experiment V
#   §4.3 Experiment Results        -> reported trends / tables / figures
#   §C.1 Implementation Details    -> 8x A100 server, Go-Explore style env reset
#
# Stages (see EXP_* below), each run with the paper's 3 random seeds
# (mean and standard deviation are reported, §4.2 Experiment I):
#
#   0. pretrain     warm-start (bottlenecked) policies pi for every task
#   1. mask         Algorithm 1: train the redesigned StateMask mask network
#                   with the fixed sample budgets of Table 4 (also the
#                   efficiency half of Experiment I: wall-clock seconds)
#   2. exp1         Experiment I: fidelity score, 500 trajectories,
#                   K = 10/20/30/40 %, 3 seeds  (§4.2 Experiment I)
#   3. exp2         Experiment II: refining effectiveness vs. No Refine,
#                   PPO fine-tuning, JSRL, StateMask-R  (§4.2 Experiment II)
#   4. exp3         Experiment III: refining with Random / StateMask / Ours
#                   explanations  (§4.2 Experiment III)
#   5. exp4         Experiment IV: non-PPO (SAC) pre-trained agent imitated
#                   with GAIL, then refined  (§4.2 Experiment IV)  [gated]
#   6. exp5         Experiment V: sweeps over p in {0,.25,.5,.75,1},
#                   lambda in {0,.1,.01,.001}, alpha in {.01,.001,.0001}
#                   (§4.2 Experiment V)
#   7. tests        pytest suite (mask net, mixed init, RND, fidelity, reset)
#
# Hyper-parameters are the paper's Table 3 values, which the scripts look up
# per task: Hopper p=.25/.001/1e-4, Walker2d p=.25/.01/1e-4,
# Reacher p=.50/.001/1e-4, HalfCheetah p=.50/.01/1e-4,
# SelfishMining p=.25/.001/1e-4, CageChallenge2 p=.50/.01/1e-4,
# Macro-v1 p=.25/.01/1e-4.  NOTE: Table 3 lists alpha = 1e-4 while §C.3's
# text says 0.01; Table 3 is operative here (see README).
#
# Out of scope (excluded on purpose, never scheduled below):
#   * all Malware Mutation experiments (Table 7 / Appendix D)
#   * SparseWalker2d refining and the sparse hyper-parameter sweeps (§C.4)
#   * autonomous-driving qualitative analysis (Figure 14 / §C.5)
#   * §3.4 theory / Appendix B proofs (no code)
#
# Usage:
#   bash scripts/run_all_experiments.sh                 # everything (SAC off)
#   bash scripts/run_all_experiments.sh --quick         # short, still < 1 GPU
#   bash scripts/run_all_experiments.sh --smoke         # tiny CPU smoke test
#   bash scripts/run_all_experiments.sh --dry-run       # print commands only
#   EXPERIMENTS="exp1 exp3" bash scripts/run_all_experiments.sh
#   TASKS="Reacher-v2" SEEDS="0 1 2" bash .../run_all_experiments.sh
#   ENABLE_SAC=1 bash .../run_all_experiments.sh --experiments exp4
# =============================================================================

set -u -o pipefail

# ----------------------------------------------------------------------------
# 0. Paths and bootstrap (tolerates both repository layouts)
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # <repo>/rice
PKG_ROOT="${REPO_ROOT}/rice"                        # <repo>/rice/rice
if [ ! -d "${PKG_ROOT}/algorithms" ] && [ -d "${REPO_ROOT}/algorithms" ]; then
    # script lives in <repo>/rice/rice/scripts
    PKG_ROOT="${REPO_ROOT}"
    REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
fi

export PYTHONPATH="${REPO_ROOT}:${PKG_ROOT}:${PYTHONPATH:-}"

# Locate an executable: $1 = python script name (e.g. train_mask.py)
resolve_script() {
    local name="$1"
    for cand in "${SCRIPT_DIR}/${name}" \
                "${REPO_ROOT}/scripts/${name}" \
                "${PKG_ROOT}/scripts/${name}" \
                "${REPO_ROOT}/rice/scripts/${name}"; do
        if [ -f "${cand}" ]; then printf '%s' "${cand}"; return 0; fi
    done
    printf ''   # not found
}

MAIN_PY=""
for cand in "${REPO_ROOT}/main.py" "${PKG_ROOT}/main.py" "${SCRIPT_DIR}/../main.py"; do
    if [ -f "${cand}" ]; then MAIN_PY="${cand}"; break; fi
done

# Python interpreter: prefer $PYTHON, then python3, then python
PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    if command -v python3 >/dev/null 2>&1; then PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then PYTHON="python"
    else echo "ERROR: no python interpreter found" >&2; exit 127; fi
fi

# ----------------------------------------------------------------------------
# 1. Configuration (env vars / CLI flags)
# ----------------------------------------------------------------------------
# Paper tasks: 4 dense MuJoCo + 3 real-world apps (§4.1) + 2 in-scope sparse
# games (§4.1 "three sparse MuJoCo games", SparseWalker2d out of scope).
DENSE_TASKS_DEFAULT="Hopper-v3 Walker2d-v3 Reacher-v2 HalfCheetah-v3 SelfishMining CageChallenge2 Macro-v1"
SPARSE_TASKS_DEFAULT="SparseHopper SparseHalfCheetah"        # SparseWalker2d OUT OF SCOPE
TASKS="${TASKS:-$DENSE_TASKS_DEFAULT $SPARSE_TASKS_DEFAULT}"

SEEDS="${SEEDS:-0 1 2}"                                       # §4.2: 3 random seeds
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs}"
EXPERIMENTS="${EXPERIMENTS:-pretrain mask exp1 exp2 exp3 exp5 tests}"
ENABLE_SAC="${ENABLE_SAC:-0}"                                 # Experiment IV gate
STRICT="${STRICT:-0}"                                         # 1 = abort on first failure
PLOT="${PLOT:-1}"
DRY_RUN="${DRY_RUN:-0}"
QUICK="${QUICK:-0}"
SMOKE="${SMOKE:-0}"

# Pre-training / refining / fidelity budgets (unspecified by the paper ->
# documented defaults; see README "Deviations").
PRETRAIN_TIMESTEPS="${PRETRAIN_TIMESTEPS:-1000000}"
REFINE_ITERATIONS="${REFINE_ITERATIONS:-100}"
FIDELITY_TRAJECTORIES="${FIDELITY_TRAJECTORIES:-500}"         # §4.2 Experiment I: 500 trajectories
FIDELITY_KS="${FIDELITY_KS:-0.1 0.2 0.3 0.4}"                 # §4.2: K = 10/20/30/40 %
SWEEP_SEEDS="${SWEEP_SEEDS:-$SEEDS}"
SAC_TIMESTEPS="${SAC_TIMESTEPS:-1000000}"
GAIL_TIMESTEPS="${GAIL_TIMESTEPS:-300000}"

EXTRA_PRETRAIN_ARGS="${EXTRA_PRETRAIN_ARGS:-}"
EXTRA_MASK_ARGS="${EXTRA_MASK_ARGS:-}"
EXTRA_FIDELITY_ARGS="${EXTRA_FIDELITY_ARGS:-}"
EXTRA_REFINE_ARGS="${EXTRA_REFINE_ARGS:-}"
EXTRA_SWEEP_ARGS="${EXTRA_SWEEP_ARGS:-}"

usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Flags:
  --experiments "a b c"   stages to run (pretrain mask exp1 exp2 exp3 exp4 exp5 tests all)
  --tasks "t1 t2"         task subset (default: all in-scope tasks)
  --seeds "0 1 2"         seeds (default: 3, per §4.2)
  --out-dir DIR           artifact directory (default: <repo>/runs)
  --device cpu|cuda|auto  torch device
  --quick                 reduced budgets (keeps every stage runnable)
  --smoke                 tiny CPU smoke run (no SAC, few trajectories)
  --dry-run               print the commands without executing them
  --strict                abort on the first failing stage (default: warn & continue)
  --no-plot               skip matplotlib figure generation
  --enable-sac            also run Experiment IV (SAC pre-train + GAIL; expensive)
  -h | --help             this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --tasks)       TASKS="$2";       shift 2 ;;
        --seeds)       SEEDS="$2"; SWEEP_SEEDS="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --pretrain-timesteps) PRETRAIN_TIMESTEPS="$2"; shift 2 ;;
        --iterations)  REFINE_ITERATIONS="$2"; shift 2 ;;
        --trajectories) FIDELITY_TRAJECTORIES="$2"; shift 2 ;;
        --quick)       QUICK=1; shift ;;
        --smoke)       SMOKE=1; QUICK=1; ENABLE_SAC=0; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --strict)      STRICT=1; shift ;;
        --no-plot)     PLOT=0; shift ;;
        --enable-sac)  ENABLE_SAC=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

RUN_SAC=0
case "${ENABLE_SAC}" in 1|true|TRUE|yes|YES|on|ON) RUN_SAC=1 ;; esac

# Budget scaling for --quick / --smoke
if [ "${SMOKE}" = "1" ]; then
    PRETRAIN_TIMESTEPS="2000"
    REFINE_ITERATIONS="3"
    FIDELITY_TRAJECTORIES="8"
    FIDELITY_KS="0.1 0.2"
    SEEDS="0"; SWEEP_SEEDS="0"
    EXTRA_MASK_ARGS="${EXTRA_MASK_ARGS} --samples 2000 --iterations 2"
elif [ "${QUICK}" = "1" ]; then
    PRETRAIN_TIMESTEPS="${PRETRAIN_TIMESTEPS_QUICK:-50000}"
    REFINE_ITERATIONS="${REFINE_ITERATIONS_QUICK:-20}"
    FIDELITY_TRAJECTORIES="${FIDELITY_TRAJECTORIES_QUICK:-50}"
    [ -z "${SWEEP_SEEDS:-}" ] && SWEEP_SEEDS="0"
fi

# ----------------------------------------------------------------------------
# 2. Logging / bookkeeping helpers
# ----------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run_all_experiments.log"
SUMMARY_TSV="${OUT_DIR}/run_all_experiments_stages.tsv"
: > "${SUMMARY_TSV}"

RUN_START="${SECONDS}"
STAGE_OK=0; STAGE_FAIL=0; STAGE_SKIP=0

log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${LOG_FILE}"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${LOG_FILE}" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${LOG_FILE}" >&2; exit 1; }

record() {   # label, status, seconds, command
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "${SUMMARY_TSV}"
    case "$2" in
        ok*)     STAGE_OK=$((STAGE_OK + 1)) ;;
        skipped) STAGE_SKIP=$((STAGE_SKIP + 1)) ;;
        *)       STAGE_FAIL=$((STAGE_FAIL + 1)) ;;
    esac
}

# run_step LABEL CMD...   -- executes, logs, records; honours DRY_RUN/STRICT
run_step() {
    local label="$1"; shift
    local start="${SECONDS}"
    log "==> ${label}: $*"
    if [ "${DRY_RUN}" = "1" ]; then
        record "${label}" "dry-run" "0" "$*"
        return 0
    fi
    if "$@" >>"${LOG_FILE}" 2>&1; then
        local dt=$((SECONDS - start))
        log "<== ${label} OK (${dt}s)"
        record "${label}" "ok" "${dt}" "$*"
        return 0
    fi
    local rc=$?
    local dt=$((SECONDS - start))
    warn "${label} failed (exit ${rc}, ${dt}s) -- see ${LOG_FILE}"
    record "${label}" "failed:${rc}" "${dt}" "$*"
    if [ "${STRICT}" = "1" ]; then
        die "aborting because --strict was given (${label})"
    fi
    return ${rc}
}

# run_script LABEL SCRIPT_PATH ARGS...   -- retries with no extra args when the
# requested CLI flags are unsupported (tolerant to script-flag drift).
run_script() {
    local label="$1" script="$2"; shift 2
    run_step "${label}" "${PYTHON}" "${script}" "$@" && return 0
    if [ "$#" -gt 0 ] && [ "${DRY_RUN}" != "1" ] && [ "${STRICT}" != "1" ]; then
        warn "retrying ${label} without optional flags"
        run_step "${label} (retry, defaults)" "${PYTHON}" "${script}"
        return $?
    fi
    return 1
}

# Run one python script for every task in $TASKS (kept sequential: RICE's
# roll-in restores a single simulator state per refining iteration).
for_each_task_script() {
    local label_prefix="$1" script="$2"; shift 2
    [ -n "${script}" ] || { warn "script not found for ${label_prefix}"; return 1; }
    local task rc=0
    for task in ${TASKS}; do
        run_script "${label_prefix}:${task}" "${script}" --task "${task}" "$@" || rc=$?
    done
    return ${rc}
}

require_script() {   # resolve_script but fail loudly if missing
    local path; path="$(resolve_script "$1")"
    if [ -z "${path}" ]; then warn "missing script: $1"; return 1; fi
    printf '%s' "${path}"
}

have_experiment() {   # have_experiment exp2 -> 0 when scheduled
    for e in ${EXPERIMENTS}; do
        [ "$e" = "$1" ] && return 0
        [ "$e" = "all" ] && return 0
    done
    return 1
}

# ----------------------------------------------------------------------------
# 3. Stage 0 -- warm-start pre-training (Phase 1 of the plan)
#    The paper's agents sit at a "bottleneck" (Assumption 3.2); script
#    pretrain_agent.py stops early once the plateau/target is reached.
# ----------------------------------------------------------------------------
stage_pretrain() {
    local script; script="$(require_script pretrain_agent.py)" || { record pretrain skipped 0 ""; return 0; }
    for_each_task_script "pretrain" "${script}" \
        --seeds ${SEEDS} --device "${DEVICE}" --out-dir "${OUT_DIR}/pretrain" \
        --timesteps "${PRETRAIN_TIMESTEPS}" ${EXTRA_PRETRAIN_ARGS}
}

# ----------------------------------------------------------------------------
# 4. Stage 1 -- Algorithm 1 mask network + Table 4 efficiency timing
#    train_mask.py uses the Table 4 sample budgets (Hopper/Walker2d/Reacher/
#    HalfCheetah 3e5, Selfish 1.5e6, Cage 1e7, Auto 2443260) and Table 3 alpha.
# ----------------------------------------------------------------------------
stage_mask() {
    local script; script="$(require_script train_mask.py)" || { record mask skipped 0 ""; return 0; }
    local mask_args="--seeds ${SEEDS} --device ${DEVICE} --out-dir ${OUT_DIR}/mask --weights ${OUT_DIR}/pretrain/weights ${EXTRA_MASK_ARGS}"
    for task in ${TASKS}; do
        run_script "mask:${task}" "${script}" --task "${task}" ${mask_args}
    done
}

# ----------------------------------------------------------------------------
# 5. Stage 2 -- Experiment I: fidelity (Figure 5) + Table 4 timing
#    500 trajectories, K in {10,20,30,40} %, 3 seeds, mean +- std (§4.2).
# ----------------------------------------------------------------------------
stage_exp1() {
    local script; script="$(require_script fidelity_eval.py)" || { record exp1 skipped 0 ""; return 0; }
    for task in ${TASKS}; do
        run_script "exp1-fidelity:${task}" "${script}" \
            --task "${task}" \
            --explanations ours statemask random integrated_gradients airs \
            --ks ${FIDELITY_KS} \
            --trajectories "${FIDELITY_TRAJECTORIES}" \
            --seeds ${SEEDS} \
            --device "${DEVICE}" \
            --out-dir "${OUT_DIR}/exp1" \
            --weights "${OUT_DIR}/pretrain/weights" \
            --mask-weights "${OUT_DIR}/mask/weights" \
            ${EXTRA_FIDELITY_ARGS}
        # random-explanation control (unbiased window baseline, §4.1)
        run_script "exp1-random:${task}" "${script}" \
            --task "${task}" --explanations random \
            --ks ${FIDELITY_KS} --trajectories "${FIDELITY_TRAJECTORIES}" \
            --seeds ${SEEDS} --device "${DEVICE}" \
            --out-dir "${OUT_DIR}/exp1-random" "${EXTRA_FIDELITY_ARGS:+$EXTRA_FIDELITY_ARGS}"
    done
    # Table 4: wall-clock cost of training the mask network with a fixed
    # sample budget (average 16.8 % drop vs. StateMask, §4.3).
    for task in ${TASKS}; do
        run_script "exp1-timing:${task}" "${script}" \
            --task "${task}" --timing --seeds ${SEEDS} --device "${DEVICE}" \
            --out-dir "${OUT_DIR}/exp1-timing" "${EXTRA_FIDELITY_ARGS:+$EXTRA_FIDELITY_ARGS}"
        # alpha sweep of the explanation (Experiment V, fidelity part):
        # fidelity is expected to be insensitive to alpha (§4.3).
        run_script "exp1-alpha:${task}" "${script}" \
            --task "${task}" --alpha 0.01 0.001 0.0001 --seeds ${SEEDS} \
            --device "${DEVICE}" --out-dir "${OUT_DIR}/exp1-alpha" \
            "${EXTRA_FIDELITY_ARGS:+$EXTRA_FIDELITY_ARGS}"
    done
}

# ----------------------------------------------------------------------------
# 6. Stage 3 -- Experiment II: refining effectiveness (dense: Table 1 final
#    reward; sparse: Figure 2 refining curves).  Methods: No Refine, PPO
#    fine-tuning, JSRL, StateMask-R, Ours.  All refining methods share OUR
#    explanation (fair comparison, §4.2 Experiment II).
# ----------------------------------------------------------------------------
stage_exp2() {
    local script; script="$(require_script run_refine.py)"
    local bscript; bscript="$(require_script run_baselines.py)"
    [ -n "${script}" ] || script="${bscript}"
    [ -n "${bscript}" ] || bscript="${script}"
    [ -n "${script}" ] || { record exp2 skipped 0 ""; return 0; }

    local common="--seeds ${SEEDS} --device ${DEVICE} --iterations ${REFINE_ITERATIONS} --weights ${OUT_DIR}/pretrain/weights --mask-weights ${OUT_DIR}/mask/weights --explanations ours"
    [ "${PLOT}" = "1" ] && common="${common} --plot --curves"

    for task in ${TASKS}; do
        # run_refine.py knows Table 3 per-task p/lambda/alpha and the Table 1
        # reference trends used for the trend check.
        run_script "exp2:${task}" "${script}" \
            --task "${task}" \
            --methods no_refine ours ppo jsrl statemask_r \
            --json "${OUT_DIR}/exp2/refine_${task}.json" \
            ${common} ${EXTRA_REFINE_ARGS}
    done
    if [ -n "${bscript}" ] && [ "${bscript}" != "${script}" ]; then
        for task in ${TASKS}; do
            run_script "exp2-baselines:${task}" "${bscript}" \
                --task "${task}" \
                --methods no_refine ours ppo jsrl statemask_r \
                --json "${OUT_DIR}/exp2/baselines_${task}.json" ${common}
        done
    fi
    # SIL comparison (Table 5, secondary in-scope result)
    if have_experiment exp2; then
        for task in Hopper-v3 Walker2d-v3 Reacher-v2 HalfCheetah-v3; do
            case " ${TASKS} " in *" ${task} "*) ;; *) continue ;; esac
            run_script "exp2-sil:${task}" "${bscript:-${script}}" \
                --task "${task}" --methods sil ours \
                --json "${OUT_DIR}/exp2/sil_${task}.json" ${common}
        done
    fi
}

# ----------------------------------------------------------------------------
# 7. Stage 4 -- Experiment III: refining under different explanations
#    Fix the refiner to RICE; vary the explanation: Random, StateMask, Ours.
#    Success trend: Ours >~ StateMask > Random (the paper's strict "Ours >
#    StateMask everywhere" claim is judged insignificant and ignored).
# ----------------------------------------------------------------------------
stage_exp3() {
    local script; script="$(require_script run_baselines.py)"
    [ -n "${script}" ] || script="$(require_script run_refine.py)"
    [ -n "${script}" ] || { record exp3 skipped 0 ""; return 0; }
    local common="--seeds ${SEEDS} --device ${DEVICE} --iterations ${REFINE_ITERATIONS} --weights ${OUT_DIR}/pretrain/weights --mask-weights ${OUT_DIR}/mask/weights"
    [ "${PLOT}" = "1" ] && common="${common} --plot"
    local exp_task
    for exp_task in ${TASKS}; do
        run_script "exp3-random:${exp_task}" "${script}" \
            --task "${exp_task}" --methods ours --explanations random \
            --json "${OUT_DIR}/exp3/${exp_task}_random.json" ${common}
        run_script "exp3-statemask:${exp_task}" "${script}" \
            --task "${exp_task}" --methods ours --explanations statemask \
            --json "${OUT_DIR}/exp3/${exp_task}_statemask.json" ${common}
        run_script "exp3-ours:${exp_task}" "${script}" \
            --task "${exp_task}" --methods ours --explanations ours \
            --json "${OUT_DIR}/exp3/${exp_task}_ours.json" ${common}
    done
    # Extra explanation baselines used in Table 6 (Ours > AIRS > IG > Random)
    for exp_task in Hopper-v3 HalfCheetah-v3; do
        case " ${TASKS} " in *" ${exp_task} "*) ;; *) continue ;; esac
        run_script "exp3-airs:${exp_task}" "${script}" \
            --task "${exp_task}" --methods ours --explanations airs \
            --json "${OUT_DIR}/exp3/${exp_task}_airs.json" ${common}
        run_script "exp3-ig:${exp_task}" "${script}" \
            --task "${exp_task}" --methods ours --explanations integrated_gradients \
            --json "${OUT_DIR}/exp3/${exp_task}_integrated_gradients.json" ${common}
    done
}

# ----------------------------------------------------------------------------
# 8. Stage 5 -- Experiment IV: refining a non-PPO (SAC) pre-trained agent.
#    SAC pre-train -> GAIL imitation -> refine with Ours vs PPO-FT /
#    StateMask-R / JSRL / SAC fine-tuning.  Gated: expensive (1e6 SAC steps
#    plus 3e5 GAIL steps); enable with --enable-sac or ENABLE_SAC=1.
# ----------------------------------------------------------------------------
stage_exp4() {
    if [ "${RUN_SAC}" != "1" ]; then
        warn "Experiment IV (SAC + GAIL) disabled; pass --enable-sac / ENABLE_SAC=1 to run it"
        record exp4 "skipped" 0 "gate"
        return 0
    fi
    local script; script="$(require_script run_sac_gail.py)"
    if [ -z "${script}" ]; then script="$(require_script run_refine.py)"; fi
    [ -n "${script}" ] || { record exp4 "skipped" 0 "no script"; return 0; }
    # The paper runs Experiment IV on Hopper (Figure 3).
    local task
    for task in Hopper-v3; do
        case " ${TASKS} " in *" ${task} "*) ;; *) continue ;; esac
        run_script "exp4-sac-gail:${task}" "${script}" \
            --task "${task}" \
            --methods ours ppo statemask_r jsrl sac \
            --explanations ours \
            --seeds ${SEEDS} --device "${DEVICE}" \
            --iterations "${REFINE_ITERATIONS}" \
            --sac-timesteps "${SAC_TIMESTEPS}" \
            --gail-timesteps "${GAIL_TIMESTEPS}" \
            --out-dir "${OUT_DIR}/exp4" \
            --enable-sac --json "${OUT_DIR}/exp4/sac_gail_${task}.json" \
            ${EXTRA_REFINE_ARGS}
    done
}

# ----------------------------------------------------------------------------
# 9. Stage 6 -- Experiment V: hyper-parameter sensitivity §4.2/§4.3
#    p in {0,0.25,0.5,0.75,1}; lambda in {0,0.1,0.01,0.001};
#    alpha in {0.01,0.001,0.0001}. Sweeps are delegated to
#    rice.evaluation.hyperparam_sweep (run_sweep) so that exactly one
#    parameter differs across sweep values; falls back to `main.py sweep`.
# ----------------------------------------------------------------------------
stage_exp5() {
    log "==> exp5: hyper-parameter sweeps (p, lambda, alpha)"
    for sweep_param in p lambda alpha; do
        local task_list=""
        case "${sweep_param}" in
            p)      task_list="${TASKS}" ;;                                # all tasks
            lambda) task_list="Hopper-v3 HalfCheetah-v3 SelfishMining" ;;  # §4.3 dense + real-world
            alpha)  task_list="Hopper-v3" ;;                               # fidelity insensitivity
        esac
        for sweep_task in ${task_list}; do
            run_step "exp5-${sweep_param}:${sweep_task}" \
                "${PYTHON}" -c '
import json, os, sys
from rice.evaluation.hyperparam_sweep import run_sweep, SweepConfig

param, task, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
seeds  = [int(s) for s in sys.argv[4].split(",") if s != ""]
grids  = {"p": [0.0, 0.25, 0.5, 0.75, 1.0],
          "lambda": [0.0, 0.1, 0.01, 0.001],
          "alpha": [0.01, 0.001, 0.0001]}
cfg = SweepConfig(task=task, param=param, values=tuple(grids[param]),
                  seeds=seeds, out_dir=out_dir, plot=False)
res = run_sweep(param=param, task=task, values=grids[param], config=cfg)
try:
    payload = res.as_dict()
except Exception:
    payload = {"task": task, "param": param,
               "values": list(grids[param]), "raw": str(res)}
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "sweep_%s_%s.json" % (param, task))
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2, default=str)
verdict = res.trend_check() if hasattr(res, "trend_check") else None
if verdict:
    print("trend_check(%s, %s): %s" % (param, task, verdict))
print("best value:", res.best_value() if hasattr(res, "best_value") else "n/a")
' "${sweep_param}" "${sweep_task}" "${OUT_DIR}/exp5" "$(echo ${SWEEP_SEEDS} | tr ' ' ',')" \
                || {
                    # Fallback: the CLI sweep sub-command of main.py
                    if [ -n "${MAIN_PY}" ]; then
                        run_script "exp5-${sweep_param}:${sweep_task} (main.py)" \
                            "${MAIN_PY}" sweep --param "${sweep_param}" \
                            --task "${sweep_task}" --seeds ${SWEEP_SEEDS} \
                            --out-dir "${OUT_DIR}/exp5" ${EXTRA_SWEEP_ARGS}
                    else
                        warn "exp5-${sweep_param}:${sweep_task} unavailable (no hyperparam_sweep, no main.py)"
                        record "exp5-${sweep_param}:${sweep_task}" "skipped" 0 "no runner"
                    fi
                }
        done
    done
}

# ----------------------------------------------------------------------------
# 10. Stage 7 -- test suite (unit / behavioral tests of the plan's checklist)
# ----------------------------------------------------------------------------
stage_tests() {
    log "==> tests: pytest suite"
    local tests_dir=""
    for cand in "${PKG_ROOT}/tests" "${REPO_ROOT}/tests" "${SCRIPT_DIR}/../tests"; do
        if [ -d "${cand}" ]; then tests_dir="${cand}"; break; fi
    done
    if [ -z "${tests_dir}" ]; then
        warn "tests/ directory not found; skipping"
        record tests "skipped" 0 "no tests dir"
        return 0
    fi
    run_step "tests:pytest" "${PYTHON}" -m pytest "${tests_dir}" -q
}

# ----------------------------------------------------------------------------
# 11. Final summary
# ----------------------------------------------------------------------------
write_summary() {
    local total=$((SECONDS - RUN_START))
    local json="${OUT_DIR}/run_all_experiments_summary.json"
    local md="${OUT_DIR}/run_all_experiments_summary.md"

    {
        printf '{\n'
        printf '  "paper": "RICE: A Refining scheme for ReInForcement learning with Explanation (ICML 2024)",\n'
        printf '  "tasks": "%s",\n' "$(printf '%s' "${TASKS}" | sed 's/"/\\"/g')"
        printf '  "seeds": "%s",\n' "${SEEDS}"
        printf '  "experiments": "%s",\n' "${EXPERIMENTS}"
        printf '  "enable_sac": %s,\n' "$([ "${RUN_SAC}" = "1" ] && echo true || echo false)"
        printf '  "device": "%s",\n' "${DEVICE}"
        printf '  "out_dir": "%s",\n' "${OUT_DIR}"
        printf '  "ok_stages": %s,\n  "failed_stages": %s,\n  "skipped_stages": %s,\n' \
               "${STAGE_OK}" "${STAGE_FAIL}" "${STAGE_SKIP}"
        printf '  "wall_clock_seconds": %s,\n' "${total}"
        printf '  "log_file": "%s",\n' "${LOG_FILE}"
        printf '  "stages": [\n'
        awk -F'\t' 'BEGIN{first=1} {if(!first) printf ",\n"; first=0;
            gsub(/"/,"\\\"",$4);
            printf "    {\"stage\": \"%s\", \"status\": \"%s\", \"seconds\": %s, \"command\": \"%s\"}", $1,$2,$3,$4}
            END{printf "\n"}' "${SUMMARY_TSV}"
        printf '  ]\n}\n'
    } > "${json}"

    {
        echo "# RICE reproduction -- run summary"
        echo
        echo "- paper: RICE (ICML 2024, PMLR 235)"
        echo "- tasks: \`${TASKS}\`"
        echo "- seeds: \`${SEEDS}\`  (Experiments I-V report mean +- std, §4.2)"
        echo "- experiments: \`${EXPERIMENTS}\`"
        echo "- Experiment IV (SAC+GAIL) enabled: ${RUN_SAC}"
        echo "- device: \`${DEVICE}\`, artifacts: \`${OUT_DIR}\`"
        echo "- wall clock: ${total}s; ok=${STAGE_OK} failed=${STAGE_FAIL} skipped=${STAGE_SKIP}"
        echo
        echo "| stage | status | seconds |"
        echo "| --- | --- | --- |"
        awk -F'\t' '{printf "| %s | %s | %s |\n", $1, $2, $3}' "${SUMMARY_TSV}"
        echo
        echo "Table 3 hyper-parameters used (per task, via the scripts):"
        echo "Hopper {p .25, lambda .001, alpha 1e-4}, Walker2d {p .25, lambda .01, alpha 1e-4},"
        echo "Reacher {p .50, lambda .001, alpha 1e-4}, HalfCheetah {p .50, lambda .01, alpha 1e-4},"
        echo "SelfishMining {p .25, lambda .001, alpha 1e-4}, CageChallenge2 {p .50, lambda .01, alpha 1e-4},"
        echo "Macro-v1 {p .25, lambda .01, alpha 1e-4}."
        echo
        echo "Out of scope: Malware Mutation (Table 7 / App. D), SparseWalker2d refining and"
        echo "sparse hyper-parameter sweeps (§C.4), autonomous-driving qualitative analysis (§C.5),"
        echo "§3.4 theory / Appendix B proofs."
    } > "${md}"

    log "summary written to ${json} and ${md}"
}

main() {
    log "RICE experiment orchestrator"
    log "  repo=${REPO_ROOT}  package=${PKG_ROOT}  python=${PYTHON}"
    log "  tasks: ${TASKS}"
    log "  seeds: ${SEEDS}   experiments: ${EXPERIMENTS}"
    log "  out_dir: ${OUT_DIR}  device: ${DEVICE}  sac: ${RUN_SAC}  strict: ${STRICT}"
    [ -n "${MAIN_PY}" ] && log "  main.py: ${MAIN_PY}" || warn "main.py not found (CLI fallbacks disabled)"
    if [ "${SMOKE}" = "1" ]; then log "  smoke mode: tiny CPU budgets"; 
    elif [ "${QUICK}" = "1" ]; then log "  quick mode: reduced budgets"; fi

    have_experiment pretrain && stage_pretrain
    have_experiment mask     && stage_mask
    have_experiment exp1     && stage_exp1
    have_experiment exp2     && stage_exp2
    have_experiment exp3     && stage_exp3
    have_experiment exp4     && stage_exp4
    have_experiment exp5     && stage_exp5
    have_experiment tests    && stage_tests

    write_summary

    if [ "${STAGE_FAIL}" -gt 0 ]; then
        warn "${STAGE_FAIL} stage(s) failed -- see ${LOG_FILE}"
        [ "${STRICT}" = "1" ] && exit 1
    fi
    log "done: ok=${STAGE_OK} failed=${STAGE_FAIL} skipped=${STAGE_SKIP}"
    return 0
}

main "$@"
