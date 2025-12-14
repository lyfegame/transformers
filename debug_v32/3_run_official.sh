#!/bin/bash
# Run official DeepSeek V3.2 inference with tensor saving
# This generates the ground truth tensors for comparison

set -e

# Configuration - MODIFY THESE
export CONVERTED_CKPT_PATH="./converted_ckpt"
export NUM_GPUS=40
export CONFIG="./official_inference/config_671B_v3.2.json"
export TEST_PROMPT="Hello, how are you today?"

# Debug settings
export DEBUG_SAVE_TENSORS=1
export DEBUG_OUTPUT_DIR="./saved_tensors/official"
export DEBUG_LAYER_LIMIT=3  # Only save first 3 layers to reduce storage

echo "=== Running Official DeepSeek V3.2 Inference ==="
echo "Checkpoint: $CONVERTED_CKPT_PATH"
echo "Num GPUs: $NUM_GPUS"
echo "Saving tensors to: $DEBUG_OUTPUT_DIR"
echo ""

mkdir -p $DEBUG_OUTPUT_DIR

# Create a temporary input file with our test prompt
TEMP_INPUT=$(mktemp)
echo "$TEST_PROMPT" > "$TEMP_INPUT"

# Run with tensor saving
cd official_inference
torchrun --nproc-per-node $NUM_GPUS generate.py \
    --ckpt-path "$CONVERTED_CKPT_PATH" \
    --config "$CONFIG" \
    --input-file "$TEMP_INPUT" \
    --max-new-tokens 20 \
    --temperature 0.0  # Greedy for determinism

rm "$TEMP_INPUT"

echo ""
echo "=== Official tensors saved to $DEBUG_OUTPUT_DIR ==="
ls -la $DEBUG_OUTPUT_DIR
