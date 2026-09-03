#!/usr/bin/env bash
set -e

###############################################################################
# FRE Pipeline for Kitchen Domain
# Runs Phase 1 (encoder VAE training), Phase 2 (IQL agent training),
# and Phase 3 (zero-shot evaluation) on the D4RL Kitchen domain.
#
# Usage:
#   bash fre/scripts/run_kitchen.sh [--gpu GPU_ID] [--seed SEED]
#       [--data_dir DIR] [--log_dir DIR] [--save_dir DIR] [--output_dir DIR]
###############################################################################

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
GPU=0
SEED=0
DOMAIN="kitchen"
DATA_DIR="$HOME/.d4rl"
LOG_DIR="./logs"
SAVE_DIR="./checkpoints"
OUTPUT_DIR="./results"

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
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
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Set environment
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "FRE Kitchen Pipeline"
echo "============================================================"
echo "GPU:        $GPU"
echo "Seed:       $SEED"
echo "Domain:     $DOMAIN"
echo "Data dir:   $DATA_DIR"
echo "Log dir:    $LOG_DIR"
echo "Save dir:   $SAVE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Timestamp:  $TIMESTAMP"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: Train FRE Encoder + Decoder (VAE)
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 1] Training FRE Encoder + Decoder..."
echo "--------------------------------------------"

python -m fre.main \
    --mode train_encoder \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
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
    --total_steps 200000 \
    --log_interval 1000 \
    --save_interval 20000 \
    --eval_interval 5000 \
    --epsilon 0.5 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256

echo "[Phase 1] Encoder training complete."

# ---------------------------------------------------------------------------
# Locate the latest encoder checkpoint
# ---------------------------------------------------------------------------
ENCODER_CHECKPOINT=$(ls -t "$SAVE_DIR"/"$DOMAIN"/encoder_checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$ENCODER_CHECKPOINT" ]; then
    # Fallback: try the default naming pattern
    ENCODER_CHECKPOINT="$SAVE_DIR/$DOMAIN/encoder_final.pt"
fi
echo "Using encoder checkpoint: $ENCODER_CHECKPOINT"

# ---------------------------------------------------------------------------
# Phase 2: Train IQL Agent with Frozen Encoder
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 2] Training IQL Agent..."
echo "--------------------------------"

python -m fre.main \
    --mode train_iql \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --K 32 \
    --d_latent 64 \
    --tau 0.7 \
    --beta 3.0 \
    --gamma 0.99 \
    --lr 3e-4 \
    --batch_size 256 \
    --total_steps 1000000 \
    --log_interval 1000 \
    --save_interval 50000 \
    --eval_interval 10000 \
    --target_update_rate 0.005 \
    --epsilon 0.5 \
    --sparsity 0.8 \
    --mlp_hidden_dim 256

echo "[Phase 2] IQL training complete."

# ---------------------------------------------------------------------------
# Locate the latest IQL checkpoint
# ---------------------------------------------------------------------------
IQL_CHECKPOINT=$(ls -t "$SAVE_DIR"/"$DOMAIN"/iql_checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$IQL_CHECKPOINT" ]; then
    # Fallback: try the default naming pattern
    IQL_CHECKPOINT="$SAVE_DIR/$DOMAIN/iql_final.pt"
fi
echo "Using IQL checkpoint: $IQL_CHECKPOINT"

# ---------------------------------------------------------------------------
# Phase 3: Zero-Shot Evaluation (Multi-Seed)
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 3] Zero-Shot Evaluation (Multi-Seed)..."
echo "------------------------------------------------"

python -m fre.main \
    --mode evaluate_multi \
    --domain "$DOMAIN" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --iql_checkpoint "$IQL_CHECKPOINT" \
    --K 32 \
    --num_episodes 20 \
    --max_episode_steps 280 \
    --seeds 0 1 2 3 4

echo "[Phase 3] Evaluation complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "FRE Kitchen Pipeline Complete!"
echo "============================================================"
echo "Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "IQL checkpoint:     $IQL_CHECKPOINT"
echo "Results saved to:   $OUTPUT_DIR"
echo "============================================================"