#!/usr/bin/env python3
"""
Compare tensors saved from official inference vs HF fork.

Usage:
    python compare_tensors.py \
        --official /tmp/official_tensors \
        --fork /tmp/hf_fork_tensors \
        --threshold 0.01

Output:
    Detailed comparison showing where tensors diverge.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch


def load_tensors(directory: str) -> Dict[str, dict]:
    """Load all tensor files from directory."""
    tensors = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.pt'):
            path = os.path.join(directory, filename)
            data = torch.load(path, map_location='cpu')
            # Extract name from filename (remove step prefix)
            name = data.get('name', filename.replace('.pt', ''))
            tensors[name] = data
    return tensors


def compare_tensor(official: dict, fork: dict, name: str, threshold: float = 0.01) -> dict:
    """Compare two tensors and return comparison metrics."""
    result = {
        "name": name,
        "match": True,
        "shape_match": True,
        "dtype_official": official["dtype"],
        "dtype_fork": fork["dtype"],
    }

    # Check shapes
    if official["shape"] != fork["shape"]:
        result["match"] = False
        result["shape_match"] = False
        result["shape_official"] = official["shape"]
        result["shape_fork"] = fork["shape"]
        return result

    result["shape"] = official["shape"]

    # Compare tensor values
    t_official = official["data"].float()
    t_fork = fork["data"].float()

    # Compute metrics
    diff = (t_official - t_fork).abs()
    result["max_diff"] = diff.max().item()
    result["mean_diff"] = diff.mean().item()
    result["std_diff"] = diff.std().item()

    # Relative difference (avoid div by zero)
    abs_official = t_official.abs()
    rel_diff = diff / (abs_official + 1e-8)
    result["max_rel_diff"] = rel_diff.max().item()
    result["mean_rel_diff"] = rel_diff.mean().item()

    # Check if tensors match within threshold
    if result["max_diff"] > threshold:
        result["match"] = False

    # Additional stats
    result["official_mean"] = official["mean"]
    result["official_std"] = official["std"]
    result["fork_mean"] = fork["mean"]
    result["fork_std"] = fork["std"]

    # Cosine similarity
    t_off_flat = t_official.flatten()
    t_fork_flat = t_fork.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        t_off_flat.unsqueeze(0),
        t_fork_flat.unsqueeze(0)
    ).item()
    result["cosine_similarity"] = cosine_sim

    return result


def print_comparison(result: dict):
    """Pretty print comparison result."""
    status = "MATCH" if result["match"] else "MISMATCH"
    color = "\033[92m" if result["match"] else "\033[91m"
    reset = "\033[0m"

    print(f"\n{color}[{status}]{reset} {result['name']}")

    if not result["shape_match"]:
        print(f"  Shape mismatch: official={result['shape_official']} vs fork={result['shape_fork']}")
        return

    print(f"  Shape: {result['shape']}")
    print(f"  Max diff: {result['max_diff']:.6e}")
    print(f"  Mean diff: {result['mean_diff']:.6e}")
    print(f"  Cosine similarity: {result['cosine_similarity']:.6f}")
    print(f"  Official mean/std: {result['official_mean']:.4f} / {result['official_std']:.4f}")
    print(f"  Fork mean/std: {result['fork_mean']:.4f} / {result['fork_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare tensors from official vs HF fork")
    parser.add_argument("--official", type=str, required=True, help="Directory with official tensors")
    parser.add_argument("--fork", type=str, required=True, help="Directory with HF fork tensors")
    parser.add_argument("--threshold", type=float, default=0.01, help="Max allowed difference")
    parser.add_argument("--verbose", action="store_true", help="Show all comparisons")
    args = parser.parse_args()

    print(f"Loading official tensors from: {args.official}")
    official_tensors = load_tensors(args.official)
    print(f"  Found {len(official_tensors)} tensors")

    print(f"Loading fork tensors from: {args.fork}")
    fork_tensors = load_tensors(args.fork)
    print(f"  Found {len(fork_tensors)} tensors")

    # Find common tensors
    common_names = set(official_tensors.keys()) & set(fork_tensors.keys())
    only_official = set(official_tensors.keys()) - set(fork_tensors.keys())
    only_fork = set(fork_tensors.keys()) - set(official_tensors.keys())

    print(f"\nCommon tensors: {len(common_names)}")
    if only_official:
        print(f"Only in official: {only_official}")
    if only_fork:
        print(f"Only in fork: {only_fork}")

    # Compare common tensors
    results = []
    for name in sorted(common_names):
        result = compare_tensor(official_tensors[name], fork_tensors[name], name, args.threshold)
        results.append(result)

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    matches = sum(1 for r in results if r["match"])
    mismatches = len(results) - matches

    print(f"\nSummary: {matches} matches, {mismatches} mismatches out of {len(results)} tensors")
    print(f"Threshold: {args.threshold}")

    # Show mismatches first
    print("\n--- MISMATCHES ---")
    for result in results:
        if not result["match"]:
            print_comparison(result)

    if args.verbose:
        print("\n--- MATCHES ---")
        for result in results:
            if result["match"]:
                print_comparison(result)

    # Find first divergence point
    print("\n" + "=" * 60)
    print("DIVERGENCE ANALYSIS")
    print("=" * 60)

    first_mismatch = None
    for result in results:
        if not result["match"]:
            first_mismatch = result
            break

    if first_mismatch:
        print(f"\nFirst divergence at: {first_mismatch['name']}")
        print(f"  Max difference: {first_mismatch['max_diff']:.6e}")
        print(f"  Cosine similarity: {first_mismatch['cosine_similarity']:.6f}")

        # Try to identify the root cause
        name = first_mismatch['name'].lower()
        if 'embedding' in name:
            print("\n  Root cause: Embedding layer - check vocab_size, embedding weights")
        elif 'indexer' in name:
            print("\n  Root cause: Indexer (V3.2 sparse attention) - check:")
            print("    - RoPE application (non-interleaved)")
            print("    - Hadamard transform")
            print("    - Score computation and scaling")
            print("    - Top-k selection")
        elif 'attn' in name or 'mla' in name:
            print("\n  Root cause: Attention (MLA) - check:")
            print("    - LoRA projection weights")
            print("    - RoPE application (interleaved)")
            print("    - KV cache handling")
        elif 'ffn' in name or 'moe' in name:
            print("\n  Root cause: FFN/MoE - check:")
            print("    - Expert routing")
            print("    - Gate computation")
        elif 'norm' in name:
            print("\n  Root cause: Normalization - check RMSNorm implementation")
        elif 'logits' in name:
            print("\n  Root cause: Final layer - check LM head weights")
    else:
        print("\nNo divergence found within threshold!")
        print("All tensors match - model outputs should be equivalent.")


if __name__ == "__main__":
    main()
