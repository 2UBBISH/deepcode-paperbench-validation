#!/usr/bin/env bash
set -e

# ============================================================
# FRE Pipeline for ExORL Domains (Walker and Cheetah)
# ============================================================
# This script automates the full FRE pipeline for ExORL:
#   Phase 1: Encoder VAE training
#   Phase 2: IQL agent training with frozen encoder
#   Phase 3: Zero-shot evaluation on all downstream tasks
#
# Usage:
#   bash fre/scripts/run_exorl.sh [--gpu GPU_ID] [--seed SEED]
#       [--domain DOMAIN] [--data_dir DIR] [--log_dir DIR]
#       [--save_dir DIR] [--output_dir DIR]
# ============================================================

# Default values
GPU="0"
SEED="0"
DOMAIN="exorl_walker"  # or exorl_cheetah
DATA_DIR="${HOME}/.exorl"
LOG_DIR="./logs"
SAVE_DIR="./checkpoints"
OUTPUT_DIR="./results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
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
            echo "Unknown argument: $1"
            echo "Usage: $0 [--gpu GPU_ID] [--seed SEED] [--domain DOMAIN] [--data_dir DIR] [--log_dir DIR] [--save_dir DIR] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Validate domain
if [[ "$DOMAIN" != "exorl_walker" && "$DOMAIN" != "exorl_cheetah" ]]; then
    echo "Error: DOMAIN must be 'exorl_walker' or 'exorl_cheetah', got '$DOMAIN'"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU"

# Create directories
mkdir -p "$LOG_DIR" "$SAVE_DIR" "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DOMAIN_SHORT="${DOMAIN#exorl_}"  # walker or cheetah

echo "=============================================="
echo "FRE ExORL Pipeline: $DOMAIN"
echo "=============================================="
echo "GPU:        $GPU"
echo "Seed:       $SEED"
echo "Domain:     $DOMAIN"
echo "Data dir:   $DATA_DIR"
echo "Log dir:    $LOG_DIR"
echo "Save dir:   $SAVE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Timestamp:  $TIMESTAMP"
echo "=============================================="

# ============================================================
# Phase 1: Train Encoder
# ============================================================
echo ""
echo "[Phase 1] Training FRE Encoder on $DOMAIN..."
echo "----------------------------------------------"

python -m fre.main \
    --mode train_encoder \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR/encoder_${DOMAIN_SHORT}_${TIMESTAMP}" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
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
    --lr 3e-4 \
    --batch_size 256 \
    --log_interval 1000 \
    --save_interval 20000 \
    --eval_interval 5000 \
    --epsilon 0.1 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256

echo "[Phase 1] Encoder training complete."

# Find the latest encoder checkpoint
ENCODER_CHECKPOINT=$(ls -t "$SAVE_DIR"/encoder_*_"$DOMAIN_SHORT"_*.pt 2>/dev/null | head -1)
if [ -z "$ENCODER_CHECKPOINT" ]; then
    # Try alternative naming pattern
    ENCODER_CHECKPOINT=$(ls -t "$SAVE_DIR"/encoder_*.pt 2>/dev/null | head -1)
fi

if [ -z "$ENCODER_CHECKPOINT" ]; then
    echo "Warning: Could not find encoder checkpoint. Using default path."
    ENCODER_CHECKPOINT="$SAVE_DIR/encoder_${DOMAIN_SHORT}_final.pt"
fi
echo "Using encoder checkpoint: $ENCODER_CHECKPOINT"

# ============================================================
# Phase 2: Train IQL Agent
# ============================================================
echo ""
echo "[Phase 2] Training IQL Agent on $DOMAIN..."
echo "----------------------------------------------"

python -m fre.main \
    --mode train_iql \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR/iql_${DOMAIN_SHORT}_${TIMESTAMP}" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --total_steps 1000000 \
    --batch_size 256 \
    --tau 0.7 \
    --beta 3.0 \
    --gamma 0.99 \
    --lr 3e-4 \
    --target_update_rate 0.005 \
    --K 32 \
    --log_interval 1000 \
    --save_interval 50000 \
    --eval_interval 10000 \
    --epsilon 0.1 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256

echo "[Phase 2] IQL training complete."

# Find the latest IQL checkpoint
IQL_CHECKPOINT=$(ls -t "$SAVE_DIR"/iql_*_"$DOMAIN_SHORT"_*.pt 2>/dev/null | head -1)
if [ -z "$IQL_CHECKPOINT" ]; then
    IQL_CHECKPOINT=$(ls -t "$SAVE_DIR"/iql_*.pt 2>/dev/null | head -1)
fi

if [ -z "$IQL_CHECKPOINT" ]; then
    echo "Warning: Could not find IQL checkpoint. Using default path."
    IQL_CHECKPOINT="$SAVE_DIR/iql_${DOMAIN_SHORT}_final.pt"
fi
echo "Using IQL checkpoint: $IQL_CHECKPOINT"

# ============================================================
# Phase 3: Zero-Shot Evaluation
# ============================================================
echo ""
echo "[Phase 3] Zero-Shot Evaluation on $DOMAIN..."
echo "----------------------------------------------"

python -m fre.main \
    --mode evaluate_multi \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --iql_checkpoint "$IQL_CHECKPOINT" \
    --seed "$SEED" \
    --seeds "0,1,2,3,4" \
    --K 32 \
    --num_episodes 20 \
    --max_episode_steps 1000

echo "[Phase 3] Evaluation complete."

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "FRE ExORL Pipeline Complete: $DOMAIN"
echo "=============================================="
echo "Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "IQL checkpoint:     $IQL_CHECKPOINT"
echo "Results saved to:   $OUTPUT_DIR"
echo "Logs saved to:      $LOG_DIR"
echo "=============================================="