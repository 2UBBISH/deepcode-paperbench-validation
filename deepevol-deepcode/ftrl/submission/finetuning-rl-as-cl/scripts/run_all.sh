#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh -- end-to-end reproduction driver for
#   "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
#    Mitigation Problem" (Wolczyk et al., 2024)
#
# Runs every experiment track from cheapest to most expensive:
#   toy -> robotic_sequence (Meta-World) -> montezuma -> nethack -> analysis
#
# Usage:
#   scripts/run_all.sh [options]
#
# Options:
#   --stage NAME      run only this stage
#                     (toy|robotic|robotic-sequence|montezuma|nethack|analysis|all)
#   --stages A,B,C    run an explicit ordered list of stages
#   --seed N          single seed (default 0)
#   --seeds "0 1 2"   explicit seed list (overrides --seed)
#   --num-seeds N     number of seeds starting at 0
#   --stub            force dependency-free stub environments (CPU smoke run)
#   --no-stub         force real environments (never fall back to stubs)
#   --total-steps N   override total fine-tuning steps for every stage
#   --output-dir DIR  results root (default <repo>/results)
#   --smoke-test      tiny override (fast CI-style run of every stage)
#   --skip-analysis   do not run the analysis stage
#   --skip-download   do not fetch NLD-AA shards / pre-trained checkpoints
#   --download-only   only perform the artifact downloads and exit
#   --dry-run         print the commands without executing them
#   --jobs N          parallelism hint (passed to trainers that accept it)
#   -h|--help         show this help
#
# Environment variables honoured:
#   FTRL_OUTPUT_DIR, FTRL_SEED, FTRL_TOTAL_STEPS, FTRL_STUB,
#   NLD_AA_DATA_DIR, NLD_AA_URL_BASE, NETHACK_CKPT_PATH,
#   AUTOASCEND_PATH, PYTHON
#
# The script is idempotent: every stage writes into its own results
# sub-directory and existing summaries are kept (use --force-clean to remove).
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${FTRL_OUTPUT_DIR:-${REPO_ROOT}/results}"
DATA_DIR="${NLD_AA_DATA_DIR:-${REPO_ROOT}/data/nld-aa}"
CKPT_DIR="${REPO_ROOT}/data/checkpoints"
SEED="${FTRL_SEED:-0}"
SEED_LIST=""
NUM_SEEDS=""
STUB="${FTRL_STUB:-auto}"          # auto | 1 | 0
TOTAL_STEPS="${FTRL_TOTAL_STEPS:-}"
SMOKE_TEST=0
SKIP_ANALYSIS=0
SKIP_DOWNLOAD=0
DOWNLOAD_ONLY=0
DRY_RUN=0
FORCE_CLEAN=0
JOBS="${FTRL_JOBS:-1}"
STAGE="all"
STAGES_ARG=""

# canonical stage order (cheap -> expensive)
ALL_STAGES=(toy robotic_sequence montezuma nethack analysis)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
  C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
  C_RESET=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""
fi

info()  { printf '%s[run_all]%s %s\n'  "${C_BLUE}"   "${C_RESET}" "$*"; }
ok()    { printf '%s[  ok  ]%s %s\n'  "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()  { printf '%s[ warn ]%s %s\n'  "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
err()   { printf '%s[error ]%s %s\n'  "${C_RED}"    "${C_RESET}" "$*" >&2; }
die()   { err "$*"; exit 1; }
hr()    { printf '%s\n' "----------------------------------------------------------------------"; }

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --stage)         STAGE="${2:-}"; shift 2 ;;
    --stage=*)       STAGE="${1#*=}"; shift ;;
    --stages)        STAGES_ARG="${2:-}"; shift 2 ;;
    --stages=*)      STAGES_ARG="${1#*=}"; shift ;;
    --seed)          SEED="${2:-}"; shift 2 ;;
    --seed=*)        SEED="${1#*=}"; shift ;;
    --seeds)         SEED_LIST="${2:-}"; shift 2 ;;
    --seeds=*)       SEED_LIST="${1#*=}"; shift ;;
    --num-seeds)     NUM_SEEDS="${2:-}"; shift 2 ;;
    --num-seeds=*)   NUM_SEEDS="${1#*=}"; shift ;;
    --stub)          STUB=1; shift ;;
    --no-stub)       STUB=0; shift ;;
    --total-steps)   TOTAL_STEPS="${2:-}"; shift 2 ;;
    --total-steps=*) TOTAL_STEPS="${1#*=}"; shift ;;
    --output-dir)    OUTPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir=*)  OUTPUT_DIR="${1#*=}"; shift ;;
    --data-dir)      DATA_DIR="${2:-}"; shift 2 ;;
    --data-dir=*)    DATA_DIR="${1#*=}"; shift ;;
    --jobs)          JOBS="${2:-}"; shift 2 ;;
    --jobs=*)        JOBS="${1#*=}"; shift ;;
    --smoke-test)    SMOKE_TEST=1; shift ;;
    --skip-analysis) SKIP_ANALYSIS=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --force-clean)   FORCE_CLEAN=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "unknown option: $1 (try --help)" ;;
  esac
done

export FTRL_OUTPUT_DIR="${OUTPUT_DIR}"
export NLD_AA_DATA_DIR="${DATA_DIR}"

# ---------------------------------------------------------------------------
# Resolve the seed list
# ---------------------------------------------------------------------------
if [ -n "${SEED_LIST}" ]; then
  # shellcheck disable=SC2206
  SEEDS=(${SEED_LIST})
elif [ -n "${NUM_SEEDS}" ]; then
  SEEDS=()
  i=0
  while [ "${i}" -lt "${NUM_SEEDS}" ]; do SEEDS+=("${i}"); i=$((i + 1)); done
else
  SEEDS=("${SEED}")
fi
export FTRL_SEEDS="${SEEDS[*]}"

# ---------------------------------------------------------------------------
# Resolve the stage list
# ---------------------------------------------------------------------------
normalise_stage() {
  case "$1" in
    toy|toy_examples|appendix_a)          echo "toy" ;;
    robotic|robotic_sequence|metaworld|meta-world|robotic-sequence) echo "robotic_sequence" ;;
    montezuma|montezumas|m1m2)            echo "montezuma" ;;
    nethack|nle)                          echo "nethack" ;;
    analysis|analyze|plots|figures)       echo "analysis" ;;
    all)                                  echo "all" ;;
    *)                                    echo "" ;;
  esac
}

SELECTED=()
if [ -n "${STAGES_ARG}" ]; then
  IFS=',' read -r -a _raw <<< "${STAGES_ARG}"
  for s in "${_raw[@]}"; do
    s="$(echo "${s}" | tr -d '[:space:]')"
    [ -z "${s}" ] && continue
    n="$(normalise_stage "${s}")"
    [ -z "${n}" ] && die "unknown stage: ${s}"
    SELECTED+=("${n}")
  done
else
  n="$(normalise_stage "${STAGE}")"
  [ -z "${n}" ] && die "unknown stage: ${STAGE} (try --help)"
  if [ "${n}" = "all" ]; then
    SELECTED=("${ALL_STAGES[@]}")
  else
    SELECTED=("${n}")
  fi
fi

if [ "${SKIP_ANALYSIS}" -eq 1 ]; then
  _keep=()
  for s in "${SELECTED[@]}"; do [ "${s}" = "analysis" ] || _keep+=("${s}"); done
  SELECTED=("${_keep[@]}")
fi

# ---------------------------------------------------------------------------
# Smoke-test overrides (fast, CPU-only)
# ---------------------------------------------------------------------------
SMOKE_ARGS=()
if [ "${SMOKE_TEST}" -eq 1 ]; then
  STUB=1
  TOTAL_STEPS="${TOTAL_STEPS:-4096}"
  NUM_SEEDS="${NUM_SEEDS:-1}"
  SEEDS=(0)
  SMOKE_ARGS+=(--smoke-test)
  info "smoke-test mode: stub environments, ${TOTAL_STEPS} steps, 1 seed"
fi

# Resolve STUB=auto -> 1 when a stub is requested, else 0 (real envs).
STUB_FLAG=()
case "${STUB}" in
  1|true|yes|True) STUB_FLAG+=(--stub) ;;
  0|false|no|False) ;;
  auto|"") ;;
  *) die "--stub/--no-stub expects a boolean" ;;
esac

TOTAL_ARGS=()
[ -n "${TOTAL_STEPS}" ] && TOTAL_ARGS+=(--total-steps "${TOTAL_STEPS}")

JOBS_ARGS=()
[ "${JOBS}" != "1" ] && JOBS_ARGS+=(--jobs "${JOBS}")

mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------
FAILURES=()

run_cmd() {
  local label="$1"; shift
  hr
  info "${C_BOLD}${label}${C_RESET}"
  info "  \$ $*"
  if [ "${DRY_RUN}" -eq 1 ]; then
    ok "dry-run (skipped)"
    return 0
  fi
  local start elapsed
  start="$(date +%s)"
  if "$@"; then
    elapsed=$(( $(date +%s) - start ))
    ok "${label} finished in ${elapsed}s"
    return 0
  fi
  elapsed=$(( $(date +%s) - start ))
  err "${label} failed after ${elapsed}s"
  FAILURES+=("${label}")
  return 1
}

# Run a command once per seed, stopping the stage on the first failure.
run_per_seed() {
  local label="$1"; shift
  local rc=0
  for s in "${SEEDS[@]}"; do
    if ! run_cmd "${label} (seed ${s})" "$@" --seed "${s}"; then
      rc=1
      break
    fi
  done
  return "${rc}"
}

check_python() {
  if ! have "${PYTHON}"; then
    die "python interpreter '${PYTHON}' not found (set \$PYTHON)"
  fi
  "${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
    || die "python >= 3.8 required"
}

check_package() {
  "${PYTHON}" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" \
    >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Stage: downloads (NLD-AA shards, pre-trained NetHack checkpoint)
# ---------------------------------------------------------------------------
stage_download() {
  hr
  info "${C_BOLD}stage: downloads${C_RESET}"
  if [ "${SKIP_DOWNLOAD}" -eq 1 ]; then
    warn "skipping downloads (--skip-download)"
    return 0
  fi

  if [ -x "${SCRIPT_DIR}/download_nld_aa.sh" ]; then
    if [ -f "${DATA_DIR}/nld-aa.db" ] || [ -f "${DATA_DIR}/nld_aa.db" ]; then
      info "NLD-AA database already present in ${DATA_DIR}; skipping download"
    else
      run_cmd "download NLD-AA shards" \
        bash "${SCRIPT_DIR}/download_nld_aa.sh" \
        --data-dir "${DATA_DIR}" --register || true
    fi
  else
    warn "scripts/download_nld_aa.sh not found - skipping dataset download"
  fi

  if [ -x "${SCRIPT_DIR}/download_pretrained_ckpt.sh" ]; then
    if [ -f "${CKPT_DIR}/nethack_challenge_30M.pt" ]; then
      info "pre-trained NetHack checkpoint already cached; skipping download"
    else
      run_cmd "download pre-trained checkpoint" \
        bash "${SCRIPT_DIR}/download_pretrained_ckpt.sh" \
        --dest-dir "${CKPT_DIR}" || true
    fi
  else
    warn "scripts/download_pretrained_ckpt.sh not found - skipping checkpoint download"
  fi
}

# ---------------------------------------------------------------------------
# Stage: toy (Appendix A: two-state MDP + AppleRetrieval)
# ---------------------------------------------------------------------------
stage_toy() {
  hr
  info "${C_BOLD}stage: toy examples (Appendix A)${C_RESET}"
  run_cmd "toy examples (both scenarios + sweeps)" \
    "${PYTHON}" -m src.toy \
    --output-dir "${OUTPUT_DIR}/toy" || true

  # Figure 9/10 reproduction through the analysis CLI.
  if check_package matplotlib; then
    run_cmd "toy: two-state MDP scenarios" \
      "${PYTHON}" -m src.toy.two_state_mdp \
      --scenario all --config configs/toy.yaml \
      --output-dir "${OUTPUT_DIR}/toy" --plot || true
    run_cmd "toy: AppleRetrieval sweeps (M and c)" \
      "${PYTHON}" -m src.toy.apple_retrieval \
      --mode both --config configs/toy.yaml \
      --output-dir "${OUTPUT_DIR}/toy/aa" --plot || true
  else
    warn "matplotlib not installed - skipping toy plots"
  fi
}

# ---------------------------------------------------------------------------
# Stage: robotic_sequence (Meta-World, SAC + EWC/BC/EM)
# ---------------------------------------------------------------------------
stage_robotic() {
  hr
  info "${C_BOLD}stage: RoboticSequence (Meta-World, SAC)${C_RESET}"
  local out="${OUTPUT_DIR}/robotic_sequence"
  mkdir -p "${out}" 2>/dev/null || true

  if check_package metaworld; then
    info "metaworld detected - running the real environments"
  else
    warn "metaworld not installed; running with --stub (Algorithm-1 mechanics only)"
    STUB_FLAG+=(--stub)
  fi

  # Pre-train pi* on the last two (FAR) stages, then fine-tune each variant.
  run_cmd "robotic: pre-train pi* (last two stages)" \
    "${PYTHON}" -m src.robotic_sequence.train_robotic \
    --config configs/robotic_sequence.yaml \
    "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" "${JOBS_ARGS[@]}" \
    --pretrain-only --output-dir "${out}/pretrain" || true

  local methods=(none ewc bc em)
  for m in "${methods[@]}"; do
    run_per_seed "robotic: fine-tune ${m}" \
      "${PYTHON}" -m src.robotic_sequence.train_robotic \
      --config configs/robotic_sequence.yaml \
      --method "${m}" --all-seeds \
      "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" "${JOBS_ARGS[@]}" \
      --output-dir "${out}/${m}" || true
  done

  run_per_seed "robotic: from-scratch baseline" \
    "${PYTHON}" -m src.robotic_sequence.train_robotic \
    --config configs/robotic_sequence.yaml --method scratch \
    "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" "${JOBS_ARGS[@]}" \
    --output-dir "${out}/scratch" || true

  run_cmd "robotic: per-stage evaluation (Figure 7)" \
    "${PYTHON}" -m evaluate --env robotic_sequence \
    --config configs/robotic_sequence.yaml \
    "${STUB_FLAG[@]}" --output-dir "${out}/eval" \
    --seeds "${SEEDS[*]}" --json || true

  run_cmd "robotic: forward transfer table (Table 6)" \
    "${PYTHON}" -m src.analysis.forward_transfer \
    --results-dir "${out}" \
    --output "${out}/forward_transfer.json" || true

  run_cmd "robotic: CKA drift analysis (Figures 26-27)" \
    "${PYTHON}" -m src.analysis.cka \
    --output-dir "${out}/cka" || true

  run_cmd "robotic: expert-action log-likelihood (Figure 8)" \
    "${PYTHON}" -m src.analysis.loglikelihood \
    --results-dir "${out}" --output-dir "${out}/loglikelihood" || true
}

# ---------------------------------------------------------------------------
# Stage: montezuma (M1 PPO+RND -> M2 BC -> fine-tuning)
# ---------------------------------------------------------------------------
stage_montezuma() {
  hr
  info "${C_BOLD}stage: Montezuma's Revenge (M1 -> M2)${C_RESET}"
  local out="${OUTPUT_DIR}/montezuma"
  mkdir -p "${out}" 2>/dev/null || true

  if ! check_package torch; then
    warn "torch not installed - skipping the Montezuma stage"
    return 0
  fi
  if ! check_package ale_py && ! check_package gymnasium && ! check_package gym; then
    warn "no Atari backend (ale-py/gym) found - running Montezuma in stub mode"
    STUB_FLAG+=(--stub)
  fi

  run_cmd "montezuma: train M1 (PPO+RND, target ~7000)" \
    "${PYTHON}" -m src.montezuma.m1_train \
    --config configs/montezuma.yaml \
    "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
    --output-dir "${out}/m1" || true

  local m1_ckpt="${out}/m1/m1.pt"
  local ckpt_args=()
  [ -f "${m1_ckpt}" ] && ckpt_args+=(--checkpoint "${m1_ckpt}")

  run_cmd "montezuma: collect Room-7 trajectories + BC pre-train M2" \
    "${PYTHON}" -m src.montezuma.m2_bc --mode pipeline \
    --config configs/montezuma.yaml \
    "${ckpt_args[@]}" "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
    --output-dir "${out}/m2" || true

  for m in none bc ewc; do
    run_cmd "montezuma: fine-tune M2 (${m})" \
      "${PYTHON}" -m src.montezuma.m2_bc --mode finetune --method "${m}" \
      --config configs/montezuma.yaml \
      "${ckpt_args[@]}" "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
      --output-dir "${out}/${m}" || true
  done

  run_cmd "montezuma: from-scratch baseline" \
    "${PYTHON}" -m src.montezuma.m2_bc --mode scratch \
    --config configs/montezuma.yaml \
    "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
    --output-dir "${out}/scratch" || true

  if [ "${SMOKE_TEST}" -eq 0 ]; then
    run_cmd "montezuma: BC KL-weight sweep (Figure 13)" \
      "${PYTHON}" -m src.montezuma.m2_bc --mode sweep-kl \
      --config configs/montezuma.yaml \
      "${STUB_FLAG[@]}" --output-dir "${out}/sweep" || true
  fi

  run_cmd "montezuma: aggregate pipeline + plots (Figures 3b, 6, 17-19)" \
    "${PYTHON}" -m src.montezuma.train_montezuma \
    --config configs/montezuma.yaml \
    --methods none bc ewc scratch \
    --seeds "${SEEDS[*]}" --plot \
    "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
    --output-dir "${out}/summary" || true
}

# ---------------------------------------------------------------------------
# Stage: nethack (APPO fine-tuning + per-level AutoAscend evaluation)
# ---------------------------------------------------------------------------
stage_nethack() {
  hr
  info "${C_BOLD}stage: NetHack Human Monk (APPO)${C_RESET}"
  local out="${OUTPUT_DIR}/nethack"
  mkdir -p "${out}" 2>/dev/null || true

  if ! check_package torch; then
    warn "torch not installed - skipping the NetHack stage"
    return 0
  fi
  if ! check_package nle; then
    warn "nle not installed - running NetHack in --stub mode"
    STUB_FLAG+=(--stub)
  fi

  # Upstream artifacts: NLD-AA dataset (BC buffer / Fisher) + pi* checkpoint.
  if [ "${SKIP_DOWNLOAD}" -eq 0 ]; then
    stage_download
  fi

  local ckpt="${NETHACK_CKPT_PATH:-${CKPT_DIR}/nethack_challenge_30M.pt}"
  local ckpt_args=()
  [ -f "${ckpt}" ] && ckpt_args+=(--checkpoint "${ckpt}")
  [ -f "${ckpt}" ] || warn "pre-trained checkpoint not found at ${ckpt}; will run without it"

  run_cmd "nethack: verify pi* (expect ~5K Human Monk score)" \
    "${PYTHON}" -m src.nethack.train_nethack --mode verify \
    --config configs/nethack.yaml "${ckpt_args[@]}" \
    "${STUB_FLAG[@]}" --output-dir "${out}/verify" || true

  run_cmd "nethack: pre-train baseline head (500M steps, encoders frozen)" \
    "${PYTHON}" -m src.nethack.pretrain_baseline \
    --config configs/nethack.yaml "${ckpt_args[@]}" \
    "${STUB_FLAG[@]}" --output-dir "${out}/baseline" || true

  local methods=(scratch none ewc bc ks)
  for m in "${methods[@]}"; do
    run_cmd "nethack: fine-tune (${m})" \
      "${PYTHON}" -m src.nethack.train_nethack --mode train --method "${m}" \
      --config configs/nethack.yaml \
      "${ckpt_args[@]}" "${STUB_FLAG[@]}" "${TOTAL_ARGS[@]}" \
      --output-dir "${out}/${m}" || true
  done

  run_cmd "nethack: aggregate results + plots (Figure 3a, Table 4/5)" \
    "${PYTHON}" -m src.nethack.train_nethack --mode aggregate \
    --config configs/nethack.yaml \
    --output-dir "${out}" --plot || true
}

# ---------------------------------------------------------------------------
# Stage: analysis (Figures 5, 3, Table 6, ...)
# ---------------------------------------------------------------------------
stage_analysis() {
  hr
  info "${C_BOLD}stage: analysis / figure regeneration${C_RESET}"
  if ! check_package matplotlib; then
    warn "matplotlib not installed - analysis figures will be skipped"
    return 0
  fi

  run_cmd "analysis: level-visitation density (Figure 5)" \
    "${PYTHON}" -m src.analysis.density_plots \
    --results-dir "${OUTPUT_DIR}/nethack" \
    --output-dir "${OUTPUT_DIR}/analysis/density" --plot || true

  run_cmd "analysis: return distributions (Figure 3)" \
    "${PYTHON}" -m src.analysis.return_distribution \
    --results-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/analysis/returns" --plot || true

  run_cmd "analysis: forward transfer summary (Table 6)" \
    "${PYTHON}" -m src.analysis.forward_transfer \
    --results-dir "${OUTPUT_DIR}/robotic_sequence" \
    --output "${OUTPUT_DIR}/analysis/forward_transfer.json" || true

  run_cmd "analysis: per-method comparison figures" \
    "${PYTHON}" -m src.analysis.plotting \
    --results-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/analysis/plots" || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  hr
  info "${C_BOLD}Fine-tuning RL as continual learning - full reproduction${C_RESET}"
  info "repo        : ${REPO_ROOT}"
  info "python      : $(${PYTHON} --version 2>&1 || echo "${PYTHON}")"
  info "output dir  : ${OUTPUT_DIR}"
  info "stages      : ${SELECTED[*]}"
  info "seeds       : ${SEEDS[*]}"
  [ -n "${TOTAL_STEPS}" ] && info "total steps : ${TOTAL_STEPS}"
  [ "${STUB_FLAG[*]:-}" = "--stub" ] && info "mode        : stub (dependency-free)"
  hr

  check_python

  if [ "${DOWNLONLY:-0}" = "1" ]; then :; fi
  if [ "${DOWNLOAD_ONLY}" -eq 1 ]; then
    stage_download
    info "downloads complete (--download-only)"
    return 0
  fi

  for stage in "${SELECTED[@]}"; do
    case "${stage}" in
      toy)                stage_toy ;;
      robotic_sequence)   stage_robotic ;;
      montezuma)          stage_montezuma ;;
      nethack)            stage_nethack ;;
      analysis)           stage_analysis ;;
      *)                  warn "unknown stage '${stage}' - skipped" ;;
    esac
  done

  hr
  if [ "${#FAILURES[@]}" -eq 0 ]; then
    ok "all stages completed"
    [ "${DRY_RUN}" -eq 0 ] && info "results written to ${OUTPUT_DIR}"
    return 0
  fi

  warn "${#FAILURES[@]} command(s) failed:"
  for f in "${FAILURES[@]}"; do warn "  - ${f}"; done
  warn "unfinished curves should be reported as reproduction gaps"
  return 1
}

main "$@"
