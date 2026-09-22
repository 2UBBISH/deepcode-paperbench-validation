#!/usr/bin/env bash
# =============================================================================
# fre/scripts/run_all.sh
#
# End-to-end reproduction driver for the FRE paper
# ("Zero-Shot Reinforcement Learning via Functional Reward Encodings").
#
# It orchestrates, for every domain and every method, the two stages of the
# paper's protocol:
#
#   1. TRAIN   -> `python -m fre.main train`      (Algorithm 1: 150k encoder
#                 steps + 850k frozen-encoder IQL steps on AntMaze, 1M + 1M on
#                 ExORL / Kitchen, exactly as encoded in configs/*.yaml)
#   2. EVAL    -> `python -m fre.main eval`       (zero-shot: encode K = 32
#                 (s, eta(s)) samples -> z -> 20 episodes x 5 seeds, returns
#                 normalized to [0, 100])
#
# ... and finally aggregates the per-task JSON result files:
#
#   3. TABLES  -> `python -m fre.main reproduce-tables`  (Table 1 / Table 4)
#
# Baselines are dispatched through the same CLI (`--method gc_iql|gc_bc|opal|
# fb|sf`), which forwards to fre.baselines.{gc_iql,gc_bc,opal,fb_sf_runner}.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   bash fre/scripts/run_all.sh                      # everything, in order
#   bash fre/scripts/run_all.sh --dry-run            # print commands only
#   bash fre/scripts/run_all.sh --domains "antmaze"  # subset of domains
#   bash fre/scripts/run_all.sh --methods "fre"      # subset of methods
#   bash fre/scripts/run_all.sh --eval-only          # reuse checkpoints
#   bash fre/scripts/run_all.sh --stages eval,tables # pick stages
#   bash fre/scripts/run_all.sh --quick              # tiny smoke-test budget
#   bash fre/scripts/run_all.sh -- --seed 3          # pass args through verbatim
#
# Anything after a bare `--` is appended verbatim to every `fre.main` call.
#
# Environment variables (all optional; CLI flags take precedence):
#   PYTHON           python interpreter                 (default: python)
#   DEVICE           torch device                       (default: cuda)
#   SEEDS            comma-separated eval seeds         (default: 0,1,2,3,4)
#   DOMAINS          comma-separated domain keys        (default: all)
#   METHODS          comma-separated method keys        (default: all)
#   RUN_DIR          checkpoint root                    (default: runs)
#   RESULTS_DIR      result-JSON root                   (default: results)
#   TASK_SET         task set to evaluate               (default: all)
#   QUICK_STEPS      steps per phase when --quick       (default: 2000)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Locate the repository root (this file lives in <root>/fre/scripts/)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Make `python -m fre.main` work regardless of how the script was invoked.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# -----------------------------------------------------------------------------
# 1. Defaults (overridable via flags / environment)
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0,1,2,3,4}"
RUN_DIR="${RUN_DIR:-runs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TASK_SET="${TASK_SET:-all}"
QUICK_STEPS="${QUICK_STEPS:-2000}"

# The four evaluation domains of Table 1.  `exorl:walker` / `exorl:cheetah`
# are split because they use different datasets and different task sets.
DEFAULT_DOMAINS="antmaze,exorl:walker,exorl:cheetah,kitchen"
# FRE first (the paper's method), then the baselines of Section 5.2.
DEFAULT_METHODS="fre,gc_iql,gc_bc,opal,fb,sf"
# Ordered pipeline stages.
DEFAULT_STAGES="train,eval,tables"

DOMAINS=""
METHODS=""
STAGES=""
DRY_RUN=0
QUICK=0
EVAL_ONLY=0
SKIP_TRAIN=0
SKIP_EVAL=0
EXTRA_ARGS=()

# -----------------------------------------------------------------------------
# 2. Logging helpers
# -----------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_INFO=$'\033[1;34m'; C_WARN=$'\033[1;33m'
  C_ERR=$'\033[1;31m'; C_OK=$'\033[1;32m'
else
  C_RESET=""; C_INFO=""; C_WARN=""; C_ERR=""; C_OK=""
fi

log()   { printf '%s[run_all]%s %s\n' "${C_INFO}" "${C_RESET}" "$*"; }
warn()  { printf '%s[run_all][warn]%s %s\n' "${C_WARN}" "${C_RESET}" "$*" >&2; }
err()   { printf '%s[run_all][error]%s %s\n' "${C_ERR}" "${C_RESET}" "$*" >&2; }
ok()    { printf '%s[run_all][ok]%s %s\n' "${C_OK}" "${C_RESET}" "$*"; }
hr()    { printf '%s\n' "------------------------------------------------------------------------"; }

usage() { sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

# -----------------------------------------------------------------------------
# 3. Argument parsing
# -----------------------------------------------------------------------------
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)     usage ;;
      --dry-run)     DRY_RUN=1; shift ;;
      --quick)       QUICK=1; shift ;;
      --eval-only)   EVAL_ONLY=1; SKIP_TRAIN=1; shift ;;
      --skip-train)  SKIP_TRAIN=1; shift ;;
      --skip-eval)   SKIP_EVAL=1; shift ;;
      --domains)     DOMAINS="${2:-}"; shift 2 ;;
      --methods)     METHODS="${2:-}"; shift 2 ;;
      --stages)      STAGES="${2:-}"; shift 2 ;;
      --seeds)       SEEDS="${2:-}"; shift 2 ;;
      --run-dir)     RUN_DIR="${2:-}"; shift 2 ;;
      --results-dir) RESULTS_DIR="${2:-}"; shift 2 ;;
      --task-set)    TASK_SET="${2:-}"; shift 2 ;;
      --device)      DEVICE="${2:-}"; shift 2 ;;
      --python)      PYTHON="${2:-}"; shift 2 ;;
      --)            shift; EXTRA_ARGS=("$@"); break ;;
      *)             warn "unknown argument '$1' (ignored)"; shift ;;
    esac
  done

  [[ -n "${DOMAINS}" ]] || DOMAINS="${DEFAULT_DOMAINS}"
  [[ -n "${METHODS}" ]] || METHODS="${DEFAULT_METHODS}"
  [[ -n "${STAGES}"  ]] || STAGES="${DEFAULT_STAGES}"

  if [[ "${EVAL_ONLY}" -eq 1 ]]; then
    STAGES="eval,tables"
  fi
}

# Helper predicates over the comma-separated selection lists.
in_list() { # in_list <needle> <csv>
  local needle="$1" csv="$2" item
  IFS=',' read -r -a _items <<< "${csv}"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# 4. Command execution
# -----------------------------------------------------------------------------
RUN_COUNT=0
FAIL_COUNT=0

# run_cmd "<human label>" <cmd...>
run_cmd() {
  local label="$1"; shift
  RUN_COUNT=$((RUN_COUNT + 1))
  hr
  log "${label}"
  printf '  $'; printf ' %q' "$@"; printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  if ! "$@"; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    err "command failed (${label}) — continuing with the remaining jobs"
    return 1
  fi
  return 0
}

# -----------------------------------------------------------------------------
# 5. CLI capability probing
#
# `fre.main` exposes the subcommands train / train-all / eval /
# reproduce-tables / reproduce / info.  Flag spellings may differ slightly
# between revisions, so we probe `--help` once per subcommand and only forward
# flags the parser actually accepts.  (All probes are cached in a temp dir.)
# -----------------------------------------------------------------------------
_HELP_CACHE_DIR="$(mktemp -d 2>/dev/null || echo "/tmp/run_all_fre_$$")"
trap '[[ -d "${_HELP_CACHE_DIR}" ]] && rm -rf "${_HELP_CACHE_DIR}"' EXIT

_help_text() { # _help_text <subcommand>
  local sub="$1" cache="${_HELP_CACHE_DIR}/${sub//[:\/]/_}.help"
  if [[ ! -s "${cache}" ]]; then
    "${PYTHON}" -m fre.main "${sub}" --help >"${cache}" 2>/dev/null || true
  fi
  cat "${cache}"
}

has_flag() { # has_flag <subcommand> <flag>
  local sub="$1" flag="$2"
  _help_text "${sub}" | grep -q -- "${flag}"
}

# Append the first flag that exists among the candidates.
push_flag() { # push_flag <array-name> <subcommand> <flag1> [flag2 ...]
  local arr="$1" sub="$2"; shift 2
  local cand
  for cand in "$@"; do
    if has_flag "${sub}" "${cand}"; then
      eval "${arr}+=(\"${cand}\")"
      return 0
    fi
  done
  return 1
}

# Append "<primary> <value>" for the first existing spelling.
push_opt() { # push_opt <array-name> <subcommand> <value> <flag1> [flag2 ...]
  local arr="$1" sub="$2" value="$3"; shift 3
  local cand
  for cand in "$@"; do
    if has_flag "${sub}" "${cand}"; then
      eval "${arr}+=(\"${cand}\" \"${value}\")"
      return 0
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# 6. Domain bookkeeping
# -----------------------------------------------------------------------------
domain_base() {           # "exorl:walker" -> "exorl"
  local d="$1"
  printf '%s' "${d%%:*}"
}

domain_sub() {            # "exorl:walker" -> "walker"  ("" if none)
  local d="$1"
  if [[ "${d}" == *:* ]]; then printf '%s' "${d#*:}"; else printf '%s' ""; fi
}

domain_config() {         # config file for a domain key
  case "$(domain_base "$1")" in
    antmaze) printf '%s' "fre/configs/antmaze.yaml" ;;
    exorl)   printf '%s' "fre/configs/exorl.yaml" ;;
    kitchen) printf '%s' "fre/configs/kitchen.yaml" ;;
    *)       printf '%s' "" ;;
  esac
}

# Filesystem-safe domain slug: "exorl:walker" -> "exorl_walker"
domain_slug() { printf '%s' "${1//:/_}"; }

# Expected checkpoint / result locations produced by `fre.main`.
checkpoint_for() { printf '%s/%s/checkpoint.pt' "${RUN_DIR}" "$(domain_slug "$1")"; }
result_for()     { printf '%s/%s__%s__%s.json' "${RESULTS_DIR}" "$1" "$(domain_slug "$2")" "${TASK_SET}"; }

# -----------------------------------------------------------------------------
# 7. Stage: TRAIN
# -----------------------------------------------------------------------------
train_fre() { # train_fre <domain>
  local domain="$1" sub="${1%%:*}"
  local _sub="train"
  local args=()
  local cfg; cfg="$(domain_config "${domain}")"
  local ckpt; ckpt="$(checkpoint_for "${domain}")"

  local phase_steps=()
  if [[ "${QUICK}" -eq 1 ]]; then
    phase_steps=("--encoder-steps" "${QUICK_STEPS}" "--policy-steps" "${QUICK_STEPS}")
  else
    # Use the per-domain strided schedule baked into configs/*.yaml.
    phase_steps=()
  fi

  push_opt args "${_sub}" "${domain}" --domain || args+=("${domain}")
  [[ -n "${cfg}" ]] && push_opt args "${_sub}" "${cfg}" --config
  push_opt args "${_sub}" "${ckpt}" --save --output --checkpoint --save-path
  push_opt args "${_sub}" "${DEVICE}" --device
  push_flag args "${_sub}" --eval-after
  push_opt args "${_sub}" "$(echo "${SEEDS}" | cut -d',' -f1)" --seed
  args+=("${phase_steps[@]:-}")

  run_cmd "TRAIN FRE  domain=${domain}  (encoder phase -> frozen-encoder IQL phase)" \
    "${PYTHON}" -m fre.main train "${args[@]}" "${EXTRA_ARGS[@]:-}" \
    || warn "FRE training failed for ${domain}"
}

train_baseline() { # train_baseline <method> <domain>
  local method="$1" domain="$2"
  local _sub="train"
  local args=()
  local cfg; cfg="$(domain_config "${domain}")"
  local ckpt="${RUN_DIR}/$(domain_slug "${domain}")_${method}.pt"

  push_opt args "${_sub}" "${method}" --method
  push_opt args "${_sub}" "${domain}" --domain || args+=("${domain}")
  [[ -n "${cfg}" ]] && push_opt args "${_sub}" "${cfg}" --config
  push_opt args "${_sub}" "${ckpt}" --save --output --checkpoint --save-path
  push_opt args "${_sub}" "${DEVICE}" --device
  push_opt args "${_sub}" "$(echo "${SEEDS}" | cut -d',' -f1)" --seed
  if [[ "${QUICK}" -eq 1 ]] && has_flag "${_sub}" --steps; then
    args+=("--steps" "${QUICK_STEPS}")
  fi

  run_cmd "TRAIN ${method}  domain=${domain}" \
    "${PYTHON}" -m fre.main train "${args[@]}" "${EXTRA_ARGS[@]:-}" \
    || warn "${method} training failed for ${domain}"
}

# -----------------------------------------------------------------------------
# 8. Stage: EVAL  (zero-shot: 32 reward-annotated samples -> z -> 20 eps x 5 seeds)
# -----------------------------------------------------------------------------
eval_fre() { # eval_fre <domain>
  local domain="$1"
  local _sub="eval"
  local args=()
  local cfg; cfg="$(domain_config "${domain}")"
  local ckpt; ckpt="$(checkpoint_for "${domain}")"

  push_opt args "${_sub}" fre --method
  push_opt args "${_sub}" "${domain}" --domain || args+=("${domain}")
  [[ -n "${cfg}" ]] && push_opt args "${_sub}" "${cfg}" --config
  push_opt args "${_sub}" "${ckpt}" --checkpoint --load --model
  push_opt args "${_sub}" "${TASK_SET}" --task-set --task_set
  push_opt args "${_sub}" "${RESULTS_DIR}" --results-dir --output-dir --out-dir
  push_opt args "${_sub}" "${DEVICE}" --device

  if ! [[ -f "${ckpt}" ]]; then
    warn "no checkpoint at ${ckpt}; evaluating the (untrained) encoder/policy as-is"
  fi

  run_cmd "EVAL  FRE  domain=${domain}  task_set=${TASK_SET}  (K=32 samples, 20 episodes x 5 seeds)" \
    "${PYTHON}" -m fre.main eval "${args[@]}" "${EXTRA_ARGS[@]:-}" \
    || warn "FRE evaluation failed for ${domain}"
}

eval_baseline() { # eval_baseline <method> <domain>
  local method="$1" domain="$2"
  local _sub="eval"
  local args=()
  local cfg; cfg="$(domain_config "${domain}")"
  local ckpt="${RUN_DIR}/$(domain_slug "${domain}")_${method}.pt"

  push_opt args "${_sub}" "${method}" --method
  push_opt args "${_sub}" "${domain}" --domain || args+=("${domain}")
  [[ -n "${cfg}" ]] && push_opt args "${_sub}" "${cfg}" --config
  push_opt args "${_sub}" "${ckpt}" --checkpoint --load --model
  push_opt args "${_sub}" "${TASK_SET}" --task-set --task_set
  push_opt args "${_sub}" "${RESULTS_DIR}" --results-dir --output-dir --out-dir
  push_opt args "${_sub}" "${DEVICE}" --device

  run_cmd "EVAL  ${method}  domain=${domain}  task_set=${TASK_SET}" \
    "${PYTHON}" -m fre.main eval "${args[@]}" "${EXTRA_ARGS[@]:-}" \
    || warn "${method} evaluation failed for ${domain}"
}

# -----------------------------------------------------------------------------
# 9. Stage: TABLES  (Table 1 main results, Table 4 scaling study)
# -----------------------------------------------------------------------------
reproduce_tables() {
  local _sub="reproduce-tables"
  local args=()

  push_opt args "${_sub}" "${RESULTS_DIR}" --results-dir --input --results
  push_opt args "${_sub}" "${RESULTS_DIR}" --output-dir --out-dir
  push_flag args "${_sub}" --table1
  push_flag args "${_sub}" --table4

  run_cmd "AGGREGATE  results -> Table 1 (Table 4 / scaling study where available)" \
    "${PYTHON}" -m fre.main reproduce-tables "${args[@]}" "${EXTRA_ARGS[@]:-}" \
    || warn "table aggregation failed"
}

# -----------------------------------------------------------------------------
# 10. Pipeline
# -----------------------------------------------------------------------------
list_domains() {
  local d
  IFS=',' read -r -a _d <<< "${DOMAINS}"
  for d in "${_d[@]}"; do printf '%s\n' "${d//[[:space:]]/}"; done
}

list_methods() {
  local m
  IFS=',' read -r -a _m <<< "${METHODS}"
  for m in "${_m[@]}"; do printf '%s\n' "${m//[[:space:]]/}"; done
}

stage_enabled() { in_list "$1" "${STAGES}"; }

main() {
  parse_args "$@"

  hr
  log "FRE reproduction driver"
  log "  project root : ${PROJECT_ROOT}"
  log "  python       : ${PYTHON}"
  log "  device       : ${DEVICE}"
  log "  domains      : ${DOMAINS}"
  log "  methods      : ${METHODS}"
  log "  stages       : ${STAGES}"
  log "  seeds        : ${SEEDS}"
  log "  task set     : ${TASK_SET}"
  log "  run dir      : ${RUN_DIR}"
  log "  results dir  : ${RESULTS_DIR}"
  [[ "${QUICK}" -eq 1 ]]     && log "  quick mode   : ${QUICK_STEPS} steps/phase"
  [[ "${DRY_RUN}" -eq 1 ]]   && warn "dry run — no command will be executed"
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && log "  extra args   : ${EXTRA_ARGS[*]}"
  hr

  mkdir -p "${RUN_DIR}" "${RESULTS_DIR}"
  "${PYTHON}" -c "import sys; print('  interpreter  :', sys.executable)" 2>/dev/null || true

  # ---------------------------------------------------------------------------
  # Stage 1 — TRAIN
  # ---------------------------------------------------------------------------
  if stage_enabled "train" && [[ "${SKIP_TRAIN}" -eq 0 ]]; then
    hr; log "STAGE 1/3  TRAIN"; hr
    local domain method
    while read -r domain; do
      [[ -n "${domain}" ]] || continue
      while read -r method; do
        [[ -n "${method}" ]] || continue
        case "${method}" in
          fre) train_fre "${domain}" ;;
          gc_iql|gc_bc|opal|fb|sf) train_baseline "${method}" "${domain}" ;;
          *) warn "unknown method '${method}' — skipped" ;;
        esac
      done < <(list_methods)
    done < <(list_domains)
  elif [[ "${SKIP_TRAIN}" -eq 1 ]]; then
    log "STAGE 1/3  TRAIN — skipped"
  fi

  # ---------------------------------------------------------------------------
  # Stage 2 — EVAL
  # ---------------------------------------------------------------------------
  if stage_enabled "eval" && [[ "${SKIP_EVAL}" -eq 0 ]]; then
    hr; log "STAGE 2/3  EVAL (zero-shot)"; hr
    local domain method
    while read -r domain; do
      [[ -n "${domain}" ]] || continue
      while read -r method; do
        [[ -n "${method}" ]] || continue
        case "${method}" in
          fre) eval_fre "${domain}" ;;
          gc_iql|gc_bc|opal|fb|sf) eval_baseline "${method}" "${domain}" ;;
          *) : ;;
        esac
      done < <(list_methods)
    done < <(list_domains)
  elif [[ "${SKIP_EVAL}" -eq 1 ]]; then
    log "STAGE 2/3  EVAL — skipped"
  fi

  # ---------------------------------------------------------------------------
  # Stage 3 — TABLES
  # ---------------------------------------------------------------------------
  if stage_enabled "tables"; then
    hr; log "STAGE 3/3  TABLES (Table 1 / Table 4)"; hr
    reproduce_tables
  fi

  # ---------------------------------------------------------------------------
  # Summary
  # ---------------------------------------------------------------------------
  hr
  log "finished: ${RUN_COUNT} command(s) dispatched, ${FAIL_COUNT} failure(s)"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    find "${RESULTS_DIR}" -maxdepth 1 -name '*.json' -printf '  result: %f\n' 2>/dev/null | sort || true
  fi
  hr

  [[ "${FAIL_COUNT}" -eq 0 ]] || return 1
  return 0
}

main "$@"
