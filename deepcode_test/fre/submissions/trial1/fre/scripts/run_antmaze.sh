#!/usr/bin/env bash
# =============================================================================
# FRE: Run AntMaze Training and Evaluation
# =============================================================================
# This script runs the full FRE pipeline on the AntMaze domain:
#   Phase 1: Train encoder+decoder (VAE) on random reward functions
#   Phase 2: Train IQL agent with frozen encoder
#   Phase 3: Zero-shot evaluation on all AntMaze downstream tasks
#
# Usage:
#   bash fre/scripts/run_antmaze.sh [--gpu GPU_ID] [--seed SEED]
#
# Example:
#   bash fre/scripts/run_antmaze.sh --gpu 0 --seed 42
# =============================================================================

set -e  # Exit on error

# Default values
GPU_ID=0
SEED=0
DOMAIN="antmaze"
DATA_DIR="${HOME}/.d4rl"
LOG_DIR="./logs"
SAVE_DIR="./checkpoints"
OUTPUT_DIR="./results"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu GPU_ID] [--seed SEED] [--data_dir DIR] [--log_dir DIR] [--save_dir DIR] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=============================================="
echo "FRE AntMaze Pipeline"
echo "=============================================="
echo "GPU:        ${GPU_ID}"
echo "Seed:       ${SEED}"
echo "Domain:     ${DOMAIN}"
echo "Data dir:   ${DATA_DIR}"
echo "Log dir:    ${LOG_DIR}"
echo "Save dir:   ${SAVE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="

# Create directories
mkdir -p "${LOG_DIR}" "${SAVE_DIR}" "${OUTPUT_DIR}"

# =============================================================================
# Phase 1: Train Encoder (VAE)
# =============================================================================
echo ""
echo "=============================================="
echo "PHASE 1: Training Encoder+Decoder (VAE)"
echo "=============================================="

python -m fre.main \
    --mode train_encoder \
    --domain ${DOMAIN} \
    --data_dir ${DATA_DIR} \
    --log_dir "${LOG_DIR}/${DOMAIN}" \
    --save_dir "${SAVE_DIR}/${DOMAIN}" \
    --seed ${SEED} \
    --total_steps 200000 \
    --K 32 \
    --K_prime 32 \
    --d_embed 128 \
    --d_model 256 \
    --num_layers 2 \
    --num_heads 4 \
    --d_latent 64 \
    --num_reward_bins 64 \
    --r_max 10.0 \
    --beta_kl 0.1 \
    --lr 1e-4 \
    --batch_size 256 \
    --epsilon 0.5 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256 \
    --log_interval 500 \
    --save_interval 10000 \
    --eval_interval 5000

echo ""
echo "Phase 1 complete!"

# =============================================================================
# Phase 2: Train IQL Agent
# =============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: Training IQL Agent"
echo "=============================================="

# Find the latest encoder checkpoint
ENCODER_CHECKPOINT=$(ls -t "${SAVE_DIR}/${DOMAIN}/encoder_epoch_"*.pt 2>/dev/null | head -1)
if [ -z "${ENCODER_CHECKPOINT}" ]; then
    # Try alternative naming pattern
    ENCODER_CHECKPOINT=$(ls -t "${SAVE_DIR}/${DOMAIN}/encoder_final.pt" 2>/dev/null | head -1)
fi
if [ -z "${ENCODER_CHECKPOINT}" ]; then
    echo "ERROR: No encoder checkpoint found in ${SAVE_DIR}/${DOMAIN}/"
    echo "Please ensure Phase 1 completed successfully."
    exit 1
fi
echo "Using encoder checkpoint: ${ENCODER_CHECKPOINT}"

python -m fre.main \
    --mode train_iql \
    --domain ${DOMAIN} \
    --data_dir ${DATA_DIR} \
    --log_dir "${LOG_DIR}/${DOMAIN}" \
    --save_dir "${SAVE_DIR}/${DOMAIN}" \
    --seed ${SEED} \
    --encoder_checkpoint "${ENCODER_CHECKPOINT}" \
    --total_steps 1000000 \
    --K 32 \
    --d_latent 64 \
    --tau 0.7 \
    --beta 3.0 \
    --gamma 0.99 \
    --lr 3e-4 \
    --batch_size 256 \
    --target_update_rate 0.005 \
    --epsilon 0.5 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256 \
    --log_interval 500 \
    --save_interval 50000 \
    --eval_interval 10000

echo ""
echo "Phase 2 complete!"

# =============================================================================
# Phase 3: Zero-Shot Evaluation
# =============================================================================
echo ""
echo "=============================================="
echo "PHASE 3: Zero-Shot Evaluation"
echo "=============================================="

# Find the latest IQL checkpoint
IQL_CHECKPOINT=$(ls -t "${SAVE_DIR}/${DOMAIN}/iql_epoch_"*.pt 2>/dev/null | head -1)
if [ -z "${IQL_CHECKPOINT}" ]; then
    IQL_CHECKPOINT=$(ls -t "${SAVE_DIR}/${DOMAIN}/iql_final.pt" 2>/dev/null | head -1)
fi
if [ -z "${IQL_CHECKPOINT}" ]; then
    echo "ERROR: No IQL checkpoint found in ${SAVE_DIR}/${DOMAIN}/"
    echo "Please ensure Phase 2 completed successfully."
    exit 1
fi
echo "Using IQL checkpoint: ${IQL_CHECKPOINT}"

python -m fre.main \
    --mode evaluate_multi \
    --domain ${DOMAIN} \
    --data_dir ${DATA_DIR} \
    --output_dir "${OUTPUT_DIR}/${DOMAIN}" \
    --encoder_path "${ENCODER_CHECKPOINT}" \
    --iql_path "${IQL_CHECKPOINT}" \
    --seed ${SEED} \
    --K 32 \
    --num_episodes 20 \
    --max_episode_steps 1000 \
    --seeds "0,1,2,3,4"

echo ""
echo "Phase 3 complete!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "FRE AntMaze Pipeline Complete!"
echo "=============================================="
echo "Checkpoints: ${SAVE_DIR}/${DOMAIN}/"
echo "Logs:        ${LOG_DIR}/${DOMAIN}/"
echo "Results:     ${OUTPUT_DIR}/${DOMAIN}/"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/${DOMAIN}/evaluation_results.json"
echo "=============================================="