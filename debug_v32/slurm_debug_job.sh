#!/bin/bash
#SBATCH --job-name=v32-debug
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=logs/v32_debug_%j.out
#SBATCH --error=logs/v32_debug_%j.err

# DeepSeek V3.2 Debug Job for 40 H200 GPUs (5 nodes x 8 GPUs)
# Modify the #SBATCH directives above to match your cluster

set -e

# ============================================================
# CONFIGURATION - MODIFY THESE
# ============================================================
export HF_CKPT_PATH="/path/to/DeepSeek-V3.2-fp8"  # Your checkpoint
export WORK_DIR="/path/to/debug_v32"              # This debug directory
export HF_FORK_PATH="/path/to/your/transformers"  # Your fork

# Number of GPUs (should match SLURM allocation)
export NUM_GPUS=$((SLURM_NNODES * 8))
echo "Running on $NUM_GPUS GPUs across $SLURM_NNODES nodes"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
module load cuda/12.1  # Adjust for your cluster
source /path/to/your/conda/env/bin/activate

# Add your fork to Python path
export PYTHONPATH="$HF_FORK_PATH/src:$PYTHONPATH"

cd $WORK_DIR
mkdir -p logs saved_tensors/official saved_tensors/hf_fork

# ============================================================
# STEP 1: Quick Sanity Check (single GPU)
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1: Quick Sanity Check"
echo "============================================================"

python quick_sanity_check.py \
    --checkpoint "$HF_CKPT_PATH" \
    --dtype bfloat16

# ============================================================
# STEP 2: Setup Official Code (if not already done)
# ============================================================
if [ ! -d "official_inference" ]; then
    echo ""
    echo "============================================================"
    echo "STEP 2: Setting up official inference code"
    echo "============================================================"
    bash 1_setup_official.sh
    python 2_instrument_official.py
fi

# ============================================================
# STEP 3: Run Official Inference (Ground Truth)
# ============================================================
echo ""
echo "============================================================"
echo "STEP 3: Running Official Inference (Ground Truth)"
echo "============================================================"

export DEBUG_SAVE_TENSORS=1
export DEBUG_OUTPUT_DIR="$WORK_DIR/saved_tensors/official"
export DEBUG_LAYER_LIMIT=3

cd official_inference
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    generate.py \
    --ckpt-path "$WORK_DIR/converted_ckpt" \
    --config config_671B_v3.2.json \
    --input-file <(echo "Hello, how are you today?") \
    --max-new-tokens 20 \
    --temperature 0.0

cd $WORK_DIR

# ============================================================
# STEP 4: Run HF Fork
# ============================================================
echo ""
echo "============================================================"
echo "STEP 4: Running HuggingFace Fork"
echo "============================================================"

# Using accelerate for multi-GPU
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    4_run_hf_fork.py \
    --checkpoint "$HF_CKPT_PATH" \
    --prompt "Hello, how are you today?" \
    --output-dir "$WORK_DIR/saved_tensors/hf_fork"

# Also run with sparse attention disabled
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    4_run_hf_fork.py \
    --checkpoint "$HF_CKPT_PATH" \
    --prompt "Hello, how are you today?" \
    --output-dir "$WORK_DIR/saved_tensors/hf_fork_dense" \
    --use-sparse=false

# ============================================================
# STEP 5: Compare Tensors
# ============================================================
echo ""
echo "============================================================"
echo "STEP 5: Comparing Tensors"
echo "============================================================"

python 5_compare_tensors.py \
    --official "$WORK_DIR/saved_tensors/official" \
    --fork "$WORK_DIR/saved_tensors/hf_fork"

# Also compare dense mode
echo ""
echo "============================================================"
echo "Dense Mode Comparison"
echo "============================================================"
python 5_compare_tensors.py \
    --official "$WORK_DIR/saved_tensors/official" \
    --fork "$WORK_DIR/saved_tensors/hf_fork_dense"

echo ""
echo "============================================================"
echo "DEBUG COMPLETE"
echo "============================================================"
echo "Results saved to: $WORK_DIR/saved_tensors/"
echo "Check the comparison output above to identify divergence point"
