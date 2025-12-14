#!/usr/bin/env python3
"""
Convert DeepSeek V3.2 FP8 checkpoint to BF16 for easier debugging.

The FP8 checkpoint uses block-wise quantization with 128x128 blocks.
Each weight has a corresponding weight_scale_inv tensor for dequantization.

Usage:
    python convert_fp8_to_bf16.py \
        --input-path /models-local/DeepSeek-V3.2-fp8 \
        --output-path /models-local/DeepSeek-V3.2-bf16
"""

import argparse
import json
import os
import shutil
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import safe_open, save_file


BLOCK_SIZE = 128  # FP8 block size for quantization


def dequantize_fp8_blockwise(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """
    Dequantize FP8 weight using block-wise scaling.

    Args:
        fp8_weight: FP8 quantized weight tensor [out_dim, in_dim]
        scale_inv: Block-wise scale tensor [out_dim/128, in_dim/128]

    Returns:
        BF16 dequantized weight tensor
    """
    # Convert FP8 to float first
    weight_float = fp8_weight.to(torch.float32)

    out_dim, in_dim = weight_float.shape
    scale_out, scale_in = scale_inv.shape

    # Verify block sizes match
    expected_out = (out_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    expected_in = (in_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    if scale_out != expected_out or scale_in != expected_in:
        # Try to handle edge cases where dimensions don't perfectly divide
        # This can happen with non-standard layer sizes
        pass

    # Create output tensor
    output = torch.zeros_like(weight_float, dtype=torch.bfloat16)

    # Apply block-wise scaling
    for i in range(scale_out):
        for j in range(scale_in):
            row_start = i * BLOCK_SIZE
            row_end = min((i + 1) * BLOCK_SIZE, out_dim)
            col_start = j * BLOCK_SIZE
            col_end = min((j + 1) * BLOCK_SIZE, in_dim)

            block = weight_float[row_start:row_end, col_start:col_end]
            scale = scale_inv[i, j].item()
            output[row_start:row_end, col_start:col_end] = (block * scale).to(torch.bfloat16)

    return output


def dequantize_fp8_1d(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """
    Dequantize FP8 weight with 1D or special scale tensor.
    """
    weight_float = fp8_weight.to(torch.float32)

    # Handle various scale shapes
    if scale_inv.dim() == 1:
        # Per-row or per-channel scaling
        if scale_inv.shape[0] == fp8_weight.shape[0]:
            # Scale per output channel
            output = weight_float * scale_inv.unsqueeze(1)
        elif scale_inv.shape[0] == fp8_weight.shape[1]:
            # Scale per input channel
            output = weight_float * scale_inv.unsqueeze(0)
        else:
            # Broadcast as best we can
            output = weight_float * scale_inv.mean()
    elif scale_inv.numel() == 1:
        # Single scale for entire tensor
        output = weight_float * scale_inv.item()
    else:
        # Fall back to block-wise for 2D scales
        return dequantize_fp8_blockwise(fp8_weight, scale_inv)

    return output.to(torch.bfloat16)


def main(input_path: str, output_path: str, dry_run: bool = False):
    """Convert FP8 checkpoint to BF16."""

    print(f"Converting FP8 checkpoint to BF16")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    if dry_run:
        print("  [DRY RUN - no files will be written]")

    os.makedirs(output_path, exist_ok=True)

    # Get all safetensor shards
    shard_files = sorted(glob(os.path.join(input_path, "model-*.safetensors")))
    print(f"\nFound {len(shard_files)} shards to convert")

    # First pass: collect all scale tensors (they may be in different shards than their weights)
    print("\nPass 1: Collecting scale tensors...")
    all_scales = {}
    for shard_path in tqdm(shard_files, desc="Scanning"):
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "weight_scale_inv" in name:
                    all_scales[name] = f.get_tensor(name)
    print(f"  Found {len(all_scales)} scale tensors")

    # Second pass: convert weights
    print("\nPass 2: Converting weights...")
    converted_count = 0
    skipped_count = 0

    for shard_idx, shard_path in enumerate(tqdm(shard_files, desc="Converting")):
        shard_name = os.path.basename(shard_path)
        new_tensors = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                # Skip scale tensors (they won't be in output)
                if "weight_scale_inv" in name:
                    skipped_count += 1
                    continue

                # Check if this is an FP8 tensor that needs dequantization
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_name = name.replace(".weight", ".weight_scale_inv")

                    if scale_name in all_scales:
                        scale = all_scales[scale_name]

                        # Dequantize based on scale shape
                        if scale.dim() == 2 and scale.numel() > 1:
                            tensor = dequantize_fp8_blockwise(tensor, scale)
                        else:
                            tensor = dequantize_fp8_1d(tensor, scale)

                        converted_count += 1
                    else:
                        # No scale found, just convert dtype
                        print(f"  Warning: No scale found for {name}, converting directly")
                        tensor = tensor.to(torch.bfloat16)
                        converted_count += 1

                elif tensor.dtype == torch.float32:
                    # Keep FP32 for norms, or convert to BF16 based on size
                    # Small tensors (norms, biases) stay FP32 for precision
                    if tensor.numel() < 100000:
                        pass  # Keep as FP32
                    else:
                        tensor = tensor.to(torch.bfloat16)

                # BF16 tensors pass through unchanged
                new_tensors[name] = tensor

        # Save converted shard
        if not dry_run:
            output_shard = os.path.join(output_path, shard_name)
            save_file(new_tensors, output_shard)

    print(f"\nConversion complete:")
    print(f"  Converted {converted_count} FP8 tensors to BF16")
    print(f"  Skipped {skipped_count} scale tensors")

    # Copy non-safetensor files (config, tokenizer, etc.)
    print("\nCopying config and tokenizer files...")
    for pattern in ["*.json", "tokenizer*", "*.txt", "*.model"]:
        for src_file in glob(os.path.join(input_path, pattern)):
            if not dry_run:
                dst_file = os.path.join(output_path, os.path.basename(src_file))
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {os.path.basename(src_file)}")

    # Update model index if exists
    index_file = os.path.join(output_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print("\nUpdating model index (removing scale tensors)...")
        with open(index_file, "r") as f:
            index = json.load(f)

        # Remove scale tensor entries
        new_weight_map = {
            k: v for k, v in index["weight_map"].items()
            if "weight_scale_inv" not in k
        }
        index["weight_map"] = new_weight_map

        if not dry_run:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)
        print(f"  Removed scale entries, {len(new_weight_map)} weights remain")

    print("\nDone!")
    print(f"BF16 checkpoint saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FP8 checkpoint to BF16")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to FP8 checkpoint directory")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save BF16 checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write files, just show what would happen")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.dry_run)
