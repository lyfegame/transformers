#!/bin/bash
# Setup script for official DeepSeek V3.2 inference code
# Run this on your cluster to prepare the ground truth implementation

set -e

# Configuration - MODIFY THESE
export HF_CKPT_PATH="/path/to/DeepSeek-V3.2-fp8"  # Your downloaded checkpoint
export OFFICIAL_CODE_DIR="./official_inference"
export CONVERTED_CKPT_PATH="./converted_ckpt"
export NUM_GPUS=40  # Your H200 count

echo "=== DeepSeek V3.2 Debug Setup ==="
echo "HF Checkpoint: $HF_CKPT_PATH"
echo "Num GPUs: $NUM_GPUS"

# Step 1: Clone official inference code from HuggingFace
echo ""
echo "=== Step 1: Downloading official inference code ==="
mkdir -p $OFFICIAL_CODE_DIR
cd $OFFICIAL_CODE_DIR

# Download all inference files
for file in model.py generate.py convert.py kernel.py config_671B_v3.2.json requirements.txt; do
    echo "Downloading $file..."
    curl -sL "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/inference/$file" -o "$file"
done

# Step 2: Install dependencies
echo ""
echo "=== Step 2: Installing dependencies ==="
pip install -r requirements.txt
pip install fast-hadamard-transform  # Required for indexer

# Step 3: Convert checkpoint for tensor parallelism
echo ""
echo "=== Step 3: Converting checkpoint for MP=$NUM_GPUS ==="
python convert.py \
    --hf-ckpt-path "$HF_CKPT_PATH" \
    --save-path "$CONVERTED_CKPT_PATH" \
    --n-experts 256 \
    --model-parallel $NUM_GPUS

echo ""
echo "=== Setup Complete ==="
echo "Converted checkpoint at: $CONVERTED_CKPT_PATH"
echo ""
echo "Next step: Run 2_instrument_official.py to add tensor saving hooks"
