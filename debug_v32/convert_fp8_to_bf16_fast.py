#!/usr/bin/env python3
"""
Fast FP8 to BF16 conversion using vectorized operations.
~100x faster than the loop-based version.
"""

import argparse
import json
import os
import shutil
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import safe_open, save_file


BLOCK_SIZE = 128


def dequantize_fp8_blockwise_fast(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """
    Fast vectorized FP8 dequantization using block-wise scaling.

    Instead of Python loops, uses reshape + broadcast for ~100x speedup.
    """
    weight_float = fp8_weight.to(torch.float32)
    out_dim, in_dim = weight_float.shape
    scale_out, scale_in = scale_inv.shape

    # Pad weight to be divisible by BLOCK_SIZE
    pad_out = (BLOCK_SIZE - out_dim % BLOCK_SIZE) % BLOCK_SIZE
    pad_in = (BLOCK_SIZE - in_dim % BLOCK_SIZE) % BLOCK_SIZE

    if pad_out > 0 or pad_in > 0:
        weight_float = torch.nn.functional.pad(weight_float, (0, pad_in, 0, pad_out))

    padded_out, padded_in = weight_float.shape

    # Reshape to blocks: [out_blocks, BLOCK_SIZE, in_blocks, BLOCK_SIZE]
    weight_blocks = weight_float.view(
        padded_out // BLOCK_SIZE, BLOCK_SIZE,
        padded_in // BLOCK_SIZE, BLOCK_SIZE
    )

    # Expand scale to match: [out_blocks, 1, in_blocks, 1]
    scale_expanded = scale_inv.view(scale_out, 1, scale_in, 1)

    # Broadcast multiply
    dequantized = weight_blocks * scale_expanded

    # Reshape back
    dequantized = dequantized.view(padded_out, padded_in)

    # Remove padding
    dequantized = dequantized[:out_dim, :in_dim]

    return dequantized.to(torch.bfloat16)


def dequantize_fp8_1d(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Handle 1D or special scale tensors."""
    weight_float = fp8_weight.to(torch.float32)

    if scale_inv.dim() == 1:
        if scale_inv.shape[0] == fp8_weight.shape[0]:
            output = weight_float * scale_inv.unsqueeze(1)
        elif scale_inv.shape[0] == fp8_weight.shape[1]:
            output = weight_float * scale_inv.unsqueeze(0)
        else:
            output = weight_float * scale_inv.mean()
    elif scale_inv.numel() == 1:
        output = weight_float * scale_inv.item()
    else:
        return dequantize_fp8_blockwise_fast(fp8_weight, scale_inv)

    return output.to(torch.bfloat16)


def main(input_path: str, output_path: str, dry_run: bool = False):
    """Convert FP8 checkpoint to BF16."""

    print(f"Converting FP8 checkpoint to BF16 (FAST vectorized)")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    if dry_run:
        print("  [DRY RUN - no files will be written]")

    os.makedirs(output_path, exist_ok=True)

    # Get all safetensor shards
    shard_files = sorted(glob(os.path.join(input_path, "model-*.safetensors")))
    print(f"\nFound {len(shard_files)} shards to convert")

    # First pass: collect all scale tensors
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

                # Skip scale tensors
                if "weight_scale_inv" in name:
                    skipped_count += 1
                    continue

                # Check if FP8 tensor needs dequantization
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_name = name.replace(".weight", ".weight_scale_inv")

                    if scale_name in all_scales:
                        scale = all_scales[scale_name]

                        if scale.dim() == 2 and scale.numel() > 1:
                            tensor = dequantize_fp8_blockwise_fast(tensor, scale)
                        else:
                            tensor = dequantize_fp8_1d(tensor, scale)

                        converted_count += 1
                    else:
                        print(f"  Warning: No scale found for {name}")
                        tensor = tensor.to(torch.bfloat16)
                        converted_count += 1

                elif tensor.dtype == torch.float32:
                    # Small tensors (norms) stay FP32, large ones convert to BF16
                    if tensor.numel() >= 100000:
                        tensor = tensor.to(torch.bfloat16)

                new_tensors[name] = tensor

        # Save converted shard
        if not dry_run:
            output_shard = os.path.join(output_path, shard_name)
            save_file(new_tensors, output_shard)

    print(f"\nConversion complete:")
    print(f"  Converted {converted_count} FP8 tensors to BF16")
    print(f"  Skipped {skipped_count} scale tensors")

    # Copy config and tokenizer files
    print("\nCopying config and tokenizer files...")
    for pattern in ["*.json", "tokenizer*", "*.txt", "*.model"]:
        for src_file in glob(os.path.join(input_path, pattern)):
            if not dry_run:
                dst_file = os.path.join(output_path, os.path.basename(src_file))
                shutil.copy2(src_file, dst_file)
                print(f"  Copied {os.path.basename(src_file)}")

    # Update model index
    index_file = os.path.join(output_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print("\nUpdating model index...")
        with open(index_file, "r") as f:
            index = json.load(f)

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
    parser = argparse.ArgumentParser(description="Fast FP8 to BF16 conversion")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.dry_run)
