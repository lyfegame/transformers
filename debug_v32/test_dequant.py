#!/usr/bin/env python3
"""
Quick test to verify FP8 dequantization logic is correct.
Tests on a single shard before running full conversion.
"""

import torch
from safetensors.torch import safe_open

BLOCK_SIZE = 128


def dequantize_fp8_blockwise(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight using block-wise scaling."""
    weight_float = fp8_weight.to(torch.float32)
    out_dim, in_dim = weight_float.shape
    scale_out, scale_in = scale_inv.shape

    output = torch.zeros_like(weight_float, dtype=torch.bfloat16)

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


def main():
    ckpt_path = "/models-local/DeepSeek-V3.2-fp8"
    shard = f"{ckpt_path}/model-00001-of-000163.safetensors"

    print("Testing FP8 dequantization...")
    print(f"Reading: {shard}\n")

    with safe_open(shard, framework="pt", device="cpu") as f:
        # Test on MLP down_proj (a typical FP8 weight)
        weight_name = "model.layers.0.mlp.down_proj.weight"
        scale_name = "model.layers.0.mlp.down_proj.weight_scale_inv"

        weight = f.get_tensor(weight_name)
        scale = f.get_tensor(scale_name)

        print(f"Weight: {weight_name}")
        print(f"  Shape: {weight.shape}")
        print(f"  Dtype: {weight.dtype}")
        print(f"  Raw values (first 5): {weight.flatten()[:5]}")
        print()

        print(f"Scale: {scale_name}")
        print(f"  Shape: {scale.shape}")
        print(f"  Dtype: {scale.dtype}")
        print(f"  Values range: [{scale.min():.6f}, {scale.max():.6f}]")
        print()

        # Verify block size
        expected_scale_shape = (
            (weight.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (weight.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        print(f"Expected scale shape (block={BLOCK_SIZE}): {expected_scale_shape}")
        print(f"Actual scale shape: {tuple(scale.shape)}")
        print(f"Match: {expected_scale_shape == tuple(scale.shape)}")
        print()

        # Dequantize
        print("Dequantizing...")
        bf16_weight = dequantize_fp8_blockwise(weight, scale)

        print(f"Output:")
        print(f"  Shape: {bf16_weight.shape}")
        print(f"  Dtype: {bf16_weight.dtype}")
        print(f"  Values range: [{bf16_weight.min():.6f}, {bf16_weight.max():.6f}]")
        print(f"  Mean: {bf16_weight.float().mean():.6f}")
        print(f"  Std: {bf16_weight.float().std():.6f}")
        print()

        # Sanity check - values should be reasonable for neural network weights
        if bf16_weight.float().std() < 1e-6:
            print("WARNING: Std is very low - dequantization may be wrong!")
        elif bf16_weight.float().std() > 10:
            print("WARNING: Std is very high - dequantization may be wrong!")
        else:
            print("Values look reasonable for neural network weights")

        # Also test on a different layer type (attention)
        print("\n" + "="*60)
        print("Testing attention weight...")

        weight_name = "model.layers.0.self_attn.q_a_proj.weight"
        scale_name = "model.layers.0.self_attn.q_a_proj.weight_scale_inv"

        if weight_name in f.keys():
            weight = f.get_tensor(weight_name)
            scale = f.get_tensor(scale_name)

            print(f"\nWeight: {weight_name}")
            print(f"  Shape: {weight.shape}, Dtype: {weight.dtype}")
            print(f"Scale: {scale.shape}")

            bf16_weight = dequantize_fp8_blockwise(weight, scale)
            print(f"Dequantized: mean={bf16_weight.float().mean():.6f}, std={bf16_weight.float().std():.6f}")
        else:
            print(f"  {weight_name} not in this shard")


if __name__ == "__main__":
    main()
