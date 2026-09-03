#!/usr/bin/env bash
# =============================================================================
# RICE: Refining via Critical State Explanation
# Complete Experiment Reproduction Script
# =============================================================================
#
# This script reproduces all five experiments from the RICE paper:
#   Experiment I   – Fidelity Comparison
#   Experiment II  – Efficiency Comparison
#   Experiment III – Refining Performance (MuJoCo dense + sparse)
#   Experiment IV  – Other Applications (Selfish Mining, CAGE, Auto Driving, Malware)
#   Experiment V   – Case Study & Sensitivity Analysis
#
# Usage:
#   bash experiments/run_experiments.sh [--env ENV] [--experiment EXP] [--seed SEED] [--gpu GPU]
#
# Examples:
#   bash experiments/run_experiments.sh --env hopper --experiment all
#   bash experiments/run_experiments.sh --experiment fidelity
#   bash experiments/run_experiments.sh --env all --experiment refining
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default settings
ENV="${ENV:-all}"
EXPERIMENT="${EXPERIMENT:-all}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
N_SEEDS="${N_SEEDS:-5}"
VERBOSE="${VERBOSE:-1}"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
mkdir -p "$OUTPUT_DIR"

# Python executable
PYTHON="${PYTHON:-python}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_section() {
    echo ""
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
    echo ""
}

run_cmd() {
    log_info "Running: $*"
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        log_error "Command failed with status $status: $*"
        return $status
    fi
    return 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash experiments/run_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV            Environment: all, hopper, walker2d, reacher, halfcheetah,"
            echo "                       selfish_mining, cage, auto_driving, malware"
            echo "                       (default: all)"
            echo "  --experiment EXP     Experiment: all, fidelity, efficiency, refining,"
            echo "                       applications, sensitivity, train_agent, train_mask,"
            echo "                       refine, evaluate"
            echo "                       (default: all)"
            echo "  --seed SEED          Base random seed (default: 0)"
            echo "  --gpu GPU            GPU device ID (default: 0)"
            echo "  --n-seeds N          Number of seeds for multi-seed experiments (default: 5)"
            echo "  --verbose LEVEL      Verbosity level (default: 1)"
            echo "  --output-dir DIR     Output directory (default: ./outputs)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES="$GPU"

# =============================================================================
# Environment Lists
# =============================================================================

MUJOCO_ENVS=("hopper" "walker2d" "reacher" "halfcheetah")
MUJOCO_SPARSE_ENVS=("sparse_hopper" "sparse_walker2d" "sparse_halfcheetah")
CUSTOM_ENVS=("selfish_mining" "cage" "auto_driving" "malware")

ALL_ENVS=("${MUJOCO_ENVS[@]}" "${CUSTOM_ENVS[@]}")

get_env_list() {
    if [ "$ENV" = "all" ]; then
        echo "${ALL_ENVS[@]}"
    elif [ "$ENV" = "mujoco" ]; then
        echo "${MUJOCO_ENVS[@]}"
    elif [ "$ENV" = "custom" ]; then
        echo "${CUSTOM_ENVS[@]}"
    else
        echo "$ENV"
    fi
}

# =============================================================================
# Phase 1: Train Base Agent
# =============================================================================

train_agent() {
    local env_name="$1"
    local seed="$2"
    local output_subdir="${OUTPUT_DIR}/${env_name}/agent/seed_${seed}"

    log_section "Phase 1: Training Base Agent on ${env_name} (seed=${seed})"

    run_cmd $PYTHON experiments/train_agent.py \
        --env "$env_name" \
        --seed "$seed" \
        --output-dir "$output_subdir" \
        --verbose "$VERBOSE" \
        --device "cuda:0"

    log_info "Agent training complete for ${env_name} (seed=${seed})"
    echo "$output_subdir"
}

# =============================================================================
# Phase 2: Train Mask Network
# =============================================================================

train_mask() {
    local env_name="$1"
    local seed="$2"
    local agent_dir="$3"
    local output_subdir="${OUTPUT_DIR}/${env_name}/mask/seed_${seed}"

    # Find the agent model path
    local agent_path
    agent_path=$(find "$agent_dir" -name "final_model.zip" -o -name "best_model.zip" | head -1)
    if [ -z "$agent_path" ]; then
        log_error "No agent model found in $agent_dir"
        return 1
    fi

    log_section "Phase 2: Training Mask Network on ${env_name} (seed=${seed})"

    run_cmd $PYTHON experiments/train_mask.py \
        --env "$env_name" \
        --agent-path "$agent_path" \
        --seed "$seed" \
        --output-dir "$output_subdir" \
        --verbose "$VERBOSE" \
        --device "cuda:0" \
        --extract-explanations \
        --compute-fidelity

    log_info "Mask training complete for ${env_name} (seed=${seed})"
    echo "$output_subdir"
}

# =============================================================================
# Phase 3: Refine Agent (RICE)
# =============================================================================

refine_agent() {
    local env_name="$1"
    local seed="$2"
    local agent_dir="$3"
    local mask_dir="$4"
    local output_subdir="${OUTPUT_DIR}/${env_name}/refined/seed_${seed}"

    # Find paths
    local agent_path
    agent_path=$(find "$agent_dir" -name "final_model.zip" -o -name "best_model.zip" | head -1)

    local critical_states_path
    critical_states_path=$(find "$mask_dir" -name "critical_states.pkl" | head -1)

    if [ -z "$agent_path" ]; then
        log_error "No agent model found in $agent_dir"
        return 1
    fi
    if [ -z "$critical_states_path" ]; then
        log_error "No critical states buffer found in $mask_dir"
        return 1
    fi

    log_section "Phase 3: Refining Agent on ${env_name} (seed=${seed})"

    run_cmd $PYTHON experiments/refine.py \
        --env "$env_name" \
        --agent-path "$agent_path" \
        --critical-states-path "$critical_states_path" \
        --seed "$seed" \
        --output-dir "$output_subdir" \
        --verbose "$VERBOSE" \
        --device "cuda:0"

    log_info "Refining complete for ${env_name} (seed=${seed})"
    echo "$output_subdir"
}

# =============================================================================
# Phase 4: Evaluate and Compare
# =============================================================================

evaluate_all() {
    local env_name="$1"
    local seed="$2"
    local agent_dir="$3"
    local refined_dir="$4"
    local output_subdir="${OUTPUT_DIR}/${env_name}/evaluation/seed_${seed}"

    local agent_path
    agent_path=$(find "$agent_dir" -name "final_model.zip" -o -name "best_model.zip" | head -1)

    local refined_path
    refined_path=$(find "$refined_dir" -name "final_model.zip" -o -name "best_model.zip" | head -1)

    log_section "Phase 4: Evaluating on ${env_name} (seed=${seed})"

    local cmd_args=(
        --env "$env_name"
        --seed "$seed"
        --output-dir "$output_subdir"
        --verbose "$VERBOSE"
        --device "cuda:0"
        --n-episodes 100
    )

    if [ -n "$agent_path" ]; then
        cmd_args+=(--agent-path "$agent_path")
    fi
    if [ -n "$refined_path" ]; then
        cmd_args+=(--refined-path "$refined_path")
    fi

    run_cmd $PYTHON experiments/evaluate.py "${cmd_args[@]}"

    log_info "Evaluation complete for ${env_name} (seed=${seed})"
    echo "$output_subdir"
}

# =============================================================================
# Experiment I: Fidelity Comparison
# =============================================================================

run_fidelity_experiment() {
    log_section "Experiment I: Fidelity Comparison"

    local env_list
    read -ra env_list <<< "$(get_env_list)"

    for env_name in "${env_list[@]}"; do
        log_info "Running fidelity experiment for ${env_name}..."

        local agent_path="${OUTPUT_DIR}/${env_name}/agent/seed_0/final_model.zip"
        local mask_path="${OUTPUT_DIR}/${env_name}/mask/seed_0/mask_model.zip"

        if [ ! -f "$agent_path" ]; then
            log_error "Agent model not found: $agent_path. Train agent first."
            continue
        fi
        if [ ! -f "$mask_path" ]; then
            log_error "Mask model not found: $mask_path. Train mask first."
            continue
        fi

        run_cmd $PYTHON experiments/evaluate.py \
            --env "$env_name" \
            --experiment fidelity \
            --agent-path "$agent_path" \
            --mask-path "$mask_path" \
            --output-dir "${OUTPUT_DIR}/${env_name}/fidelity" \
            --verbose "$VERBOSE" \
            --device "cuda:0" \
            --n-episodes 100
    done

    log_info "Experiment I (Fidelity) complete."
}

# =============================================================================
# Experiment II: Efficiency Comparison
# =============================================================================

run_efficiency_experiment() {
    log_section "Experiment II: Efficiency Comparison"

    local env_list
    read -ra env_list <<< "$(get_env_list)"

    for env_name in "${env_list[@]}"; do
        log_info "Running efficiency experiment for ${env_name}..."

        run_cmd $PYTHON experiments/evaluate.py \
            --env "$env_name" \
            --experiment efficiency \
            --output-dir "${OUTPUT_DIR}/${env_name}/efficiency" \
            --verbose "$VERBOSE" \
            --device "cuda:0"
    done

    log_info "Experiment II (Efficiency) complete."
}

# =============================================================================
# Experiment III: Refining Performance (MuJoCo)
# =============================================================================

run_refining_experiment() {
    log_section "Experiment III: Refining Performance"

    local env_list=("${MUJOCO_ENVS[@]}")

    for env_name in "${env_list[@]}"; do
        log_info "Running refining experiment for ${env_name}..."

        local agent_path="${OUTPUT_DIR}/${env_name}/agent/seed_0/final_model.zip"
        local refined_path="${OUTPUT_DIR}/${env_name}/refined/seed_0/final_model.zip"

        local cmd_args=(
            --env "$env_name"
            --experiment refining
            --output-dir "${OUTPUT_DIR}/${env_name}/refining_comparison"
            --verbose "$VERBOSE"
            --device "cuda:0"
            --n-episodes 100
        )

        if [ -f "$agent_path" ]; then
            cmd_args+=(--agent-path "$agent_path")
        fi
        if [ -f "$refined_path" ]; then
            cmd_args+=(--refined-path "$refined_path")
        fi

        run_cmd $PYTHON experiments/evaluate.py "${cmd_args[@]}"
    done

    log_info "Experiment III (Refining Performance) complete."
}

# =============================================================================
# Experiment IV: Other Applications
# =============================================================================

run_applications_experiment() {
    log_section "Experiment IV: Other Applications"

    local env_list=("${CUSTOM_ENVS[@]}")

    for env_name in "${env_list[@]}"; do
        log_info "Running application experiment for ${env_name}..."

        local agent_path="${OUTPUT_DIR}/${env_name}/agent/seed_0/final_model.zip"
        local refined_path="${OUTPUT_DIR}/${env_name}/refined/seed_0/final_model.zip"

        local cmd_args=(
            --env "$env_name"
            --experiment applications
            --output-dir "${OUTPUT_DIR}/${env_name}/applications"
            --verbose "$VERBOSE"
            --device "cuda:0"
            --n-episodes 100
        )

        if [ -f "$agent_path" ]; then
            cmd_args+=(--agent-path "$agent_path")
        fi
        if [ -f "$refined_path" ]; then
            cmd_args+=(--refined-path "$refined_path")
        fi

        run_cmd $PYTHON experiments/evaluate.py "${cmd_args[@]}"
    done

    log_info "Experiment IV (Applications) complete."
}

# =============================================================================
# Experiment V: Sensitivity Analysis
# =============================================================================

run_sensitivity_experiment() {
    log_section "Experiment V: Sensitivity Analysis"

    local env_name="${1:-hopper}"

    local agent_path="${OUTPUT_DIR}/${env_name}/agent/seed_0/final_model.zip"
    local critical_states_path="${OUTPUT_DIR}/${env_name}/mask/seed_0/critical_states.pkl"

    if [ ! -f "$agent_path" ]; then
        log_error "Agent model not found: $agent_path"
        return 1
    fi
    if [ ! -f "$critical_states_path" ]; then
        log_error "Critical states not found: $critical_states_path"
        return 1
    fi

    # Sensitivity over p (mixed init probability)
    log_info "Running sensitivity analysis for p..."
    for p in 0.0 0.25 0.5 0.75 1.0; do
        run_cmd $PYTHON experiments/evaluate.py \
            --env "$env_name" \
            --experiment sensitivity \
            --agent-path "$agent_path" \
            --critical-states-path "$critical_states_path" \
            --sensitivity-param p \
            --sensitivity-value "$p" \
            --output-dir "${OUTPUT_DIR}/${env_name}/sensitivity/p_${p}" \
            --verbose "$VERBOSE" \
            --device "cuda:0" \
            --n-episodes 100
    done

    # Sensitivity over lambda (RND bonus coefficient)
    log_info "Running sensitivity analysis for lambda..."
    for lam in 0.0 0.001 0.01 0.1; do
        run_cmd $PYTHON experiments/evaluate.py \
            --env "$env_name" \
            --experiment sensitivity \
            --agent-path "$agent_path" \
            --critical-states-path "$critical_states_path" \
            --sensitivity-param lambda \
            --sensitivity-value "$lam" \
            --output-dir "${OUTPUT_DIR}/${env_name}/sensitivity/lambda_${lam}" \
            --verbose "$VERBOSE" \
            --device "cuda:0" \
            --n-episodes 100
    done

    # Sensitivity over alpha (mask intrinsic reward coefficient)
    log_info "Running sensitivity analysis for alpha..."
    for alpha in 0.01 0.001 0.0001; do
        run_cmd $PYTHON experiments/evaluate.py \
            --env "$env_name" \
            --experiment sensitivity \
            --agent-path "$agent_path" \
            --critical-states-path "$critical_states_path" \
            --sensitivity-param alpha \
            --sensitivity-value "$alpha" \
            --output-dir "${OUTPUT_DIR}/${env_name}/sensitivity/alpha_${alpha}" \
            --verbose "$VERBOSE" \
            --device "cuda:0" \
            --n-episodes 100
    done

    log_info "Experiment V (Sensitivity) complete."
}

# =============================================================================
# Full Pipeline for a Single Environment
# =============================================================================

run_full_pipeline() {
    local env_name="$1"

    log_section "Running Full Pipeline for ${env_name}"

    for seed in $(seq 0 $((N_SEEDS - 1))); do
        log_info "Processing seed ${seed} for ${env_name}..."

        # Phase 1: Train agent
        local agent_dir
        agent_dir=$(train_agent "$env_name" "$seed")

        # Phase 2: Train mask
        local mask_dir
        mask_dir=$(train_mask "$env_name" "$seed" "$agent_dir")

        # Phase 3: Refine
        local refined_dir
        refined_dir=$(refine_agent "$env_name" "$seed" "$agent_dir" "$mask_dir")

        # Phase 4: Evaluate
        evaluate_all "$env_name" "$seed" "$agent_dir" "$refined_dir"
    done

    log_info "Full pipeline complete for ${env_name}."
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_section "RICE Experiment Reproduction"
    log_info "Environment: ${ENV}"
    log_info "Experiment: ${EXPERIMENT}"
    log_info "Seed: ${SEED}"
    log_info "GPU: ${GPU}"
    log_info "N Seeds: ${N_SEEDS}"
    log_info "Output Dir: ${OUTPUT_DIR}"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    case "$EXPERIMENT" in
        all)
            # Run full pipeline for all environments
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                run_full_pipeline "$env_name"
            done

            # Run specific experiments
            run_fidelity_experiment
            run_efficiency_experiment
            run_refining_experiment
            run_applications_experiment
            run_sensitivity_experiment "hopper"
            ;;

        fidelity)
            run_fidelity_experiment
            ;;

        efficiency)
            run_efficiency_experiment
            ;;

        refining)
            run_refining_experiment
            ;;

        applications)
            run_applications_experiment
            ;;

        sensitivity)
            run_sensitivity_experiment "${ENV}"
            ;;

        train_agent)
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                train_agent "$env_name" "$SEED"
            done
            ;;

        train_mask)
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                local agent_dir="${OUTPUT_DIR}/${env_name}/agent/seed_${SEED}"
                train_mask "$env_name" "$SEED" "$agent_dir"
            done
            ;;

        refine)
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                local agent_dir="${OUTPUT_DIR}/${env_name}/agent/seed_${SEED}"
                local mask_dir="${OUTPUT_DIR}/${env_name}/mask/seed_${SEED}"
                refine_agent "$env_name" "$SEED" "$agent_dir" "$mask_dir"
            done
            ;;

        evaluate)
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                local agent_dir="${OUTPUT_DIR}/${env_name}/agent/seed_${SEED}"
                local refined_dir="${OUTPUT_DIR}/${env_name}/refined/seed_${SEED}"
                evaluate_all "$env_name" "$SEED" "$agent_dir" "$refined_dir"
            done
            ;;

        pipeline)
            # Run full pipeline for specified environment
            local env_list
            read -ra env_list <<< "$(get_env_list)"
            for env_name in "${env_list[@]}"; do
                run_full_pipeline "$env_name"
            done
            ;;

        *)
            log_error "Unknown experiment: ${EXPERIMENT}"
            log_info "Valid options: all, fidelity, efficiency, refining, applications, sensitivity, train_agent, train_mask, refine, evaluate, pipeline"
            exit 1
            ;;
    esac

    log_section "All Experiments Complete!"
    log_info "Results saved to: ${OUTPUT_DIR}"
}

# Run main
main "$@"