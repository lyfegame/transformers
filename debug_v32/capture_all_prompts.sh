#!/bin/bash
# Capture tensors for all reference prompts to separate directories
# Run this on the cluster: bash capture_all_prompts.sh

set -e

# Configuration
BASE_OUTPUT_DIR="/mnt/models-disk/official_tensors"
CKPT_PATH="/models-local/DeepSeek-V3.2-converted-mp8"
CONFIG="config_671B_v3.2.json"
LAYER_LIMIT=5

# Unset NCCL socket for single-node
unset NCCL_SOCKET_IFNAME

cd ~/deepseek-v3.2-inference

echo "=== Capturing tensors for all reference prompts ==="
echo "Base output dir: $BASE_OUTPUT_DIR"
echo ""

# Prompt 0: Simple math
echo "[Prompt 0] Simple math: What is 2+2?"
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_0_simple_math"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt "What is 2+2?"
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""

# Prompt 1: Greeting
echo "[Prompt 1] Greeting: Hello, how are you?"
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_1_greeting"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt "Hello, how are you?"
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""

# Prompt 2: Code generation
echo "[Prompt 2] Code generation: Write a Python function..."
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_2_code_generation"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt "Write a Python function to check if a number is prime."
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""

# Prompt 3: Explanation
echo "[Prompt 3] Explanation: Explain the theory of relativity..."
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_3_explanation"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt "Explain the theory of relativity in simple terms."
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""

# Prompt 4: Long context (tests sparse attention)
echo "[Prompt 4] Long context: Machine learning document..."
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_4_long_context"
mkdir -p "$OUTPUT_DIR"
LONG_PROMPT="The following is a long document about machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention. Machine learning algorithms are often categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled datasets to train algorithms to classify data or predict outcomes accurately. Unsupervised learning analyzes and clusters unlabeled datasets to discover hidden patterns without human intervention. Reinforcement learning trains algorithms through a system of reward and punishment, learning to take actions that maximize rewards.

Question: What are the three main categories of machine learning algorithms?"
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt "$LONG_PROMPT"
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""

# Prompt 5: Sparse trigger (>2048 tokens to ACTUALLY trigger sparse attention)
echo "[Prompt 5] Sparse trigger: Very long transformer document (~2500 tokens)..."
OUTPUT_DIR="$BASE_OUTPUT_DIR/prompt_5_sparse_trigger"
mkdir -p "$OUTPUT_DIR"

# Read the long prompt from file (too long for shell argument)
SPARSE_PROMPT_FILE="$(dirname "$0")/long_prompt_sparse.txt"
if [ ! -f "$SPARSE_PROMPT_FILE" ]; then
    echo "ERROR: Long prompt file not found at $SPARSE_PROMPT_FILE"
    echo "Please ensure long_prompt_sparse.txt exists in the debug_v32 directory"
    exit 1
fi

# Use --prompt-file argument if available, otherwise use stdin
# Note: capture_tensors.py needs to be updated to support --prompt-file
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --layer-limit $LAYER_LIMIT \
    --prompt-file "$SPARSE_PROMPT_FILE"
echo "  -> Saved to $OUTPUT_DIR ($(ls "$OUTPUT_DIR" | wc -l) files)"
echo ""
echo "  NOTE: This prompt should have >2048 tokens, triggering actual sparse attention!"
echo "        Check that topk_indices shape is [1, seq_len, 2048] not [1, seq_len, seq_len]"
echo ""

echo "=== All captures complete ==="
echo ""
echo "Directory structure:"
ls -la "$BASE_OUTPUT_DIR"/
echo ""
echo "Total files per directory:"
for dir in "$BASE_OUTPUT_DIR"/prompt_*; do
    echo "  $(basename $dir): $(ls "$dir" | wc -l) files"
done
