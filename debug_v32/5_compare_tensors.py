#!/usr/bin/env python3
"""
Compare tensors saved from official and HF fork implementations.

This script:
1. Loads tensors from both implementations
2. Compares them numerically
3. Reports where divergence occurs
4. Helps identify the root cause of issues

Usage:
    python 5_compare_tensors.py
    python 5_compare_tensors.py --official ./saved_tensors/official --fork ./saved_tensors/hf_fork
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import numpy as np


def load_tensors(directory: Path) -> Dict[str, torch.Tensor]:
    """Load all .pt files from a directory."""
    tensors = {}
    for pt_file in sorted(directory.glob("*.pt")):
        name = pt_file.stem
        tensors[name] = torch.load(pt_file, map_location="cpu")
    return tensors


def compare_tensors(
    name: str,
    official: torch.Tensor,
    fork: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, Dict]:
    """
    Compare two tensors and return detailed statistics.

    Returns:
        (is_close, stats_dict)
    """
    # Handle shape mismatch
    if official.shape != fork.shape:
        return False, {
            "error": "shape_mismatch",
            "official_shape": tuple(official.shape),
            "fork_shape": tuple(fork.shape),
        }

    # Convert to float for comparison
    official = official.float()
    fork = fork.float()

    # Compute differences
    abs_diff = (official - fork).abs()
    rel_diff = abs_diff / (official.abs() + 1e-10)

    # Check if close
    is_close = torch.allclose(official, fork, rtol=rtol, atol=atol)

    # Compute statistics
    stats = {
        "is_close": is_close,
        "shape": tuple(official.shape),
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "official_mean": official.mean().item(),
        "official_std": official.std().item(),
        "fork_mean": fork.mean().item(),
        "fork_std": fork.std().item(),
        "correlation": torch.corrcoef(torch.stack([
            official.flatten()[:10000],  # Limit for memory
            fork.flatten()[:10000]
        ]))[0, 1].item() if official.numel() > 1 else 1.0,
    }

    # Find location of max difference
    if abs_diff.numel() > 0:
        max_idx = abs_diff.argmax().item()
        max_coords = np.unravel_index(max_idx, abs_diff.shape)
        stats["max_diff_location"] = max_coords
        stats["official_at_max"] = official.flatten()[max_idx].item()
        stats["fork_at_max"] = fork.flatten()[max_idx].item()

    return is_close, stats


def analyze_divergence(official_tensors: Dict, fork_tensors: Dict) -> Dict:
    """
    Analyze where divergence first occurs in the model.

    Returns a structured analysis.
    """
    # Define the expected order of checkpoints
    checkpoint_order = [
        "embedding_output",
        # Layer 0
        "indexer_L0_input_x",
        "indexer_L0_input_qr",
        "indexer_L0_q_after_rope",
        "indexer_L0_k_after_rope",
        "indexer_L0_q_after_hadamard",
        "indexer_L0_k_after_hadamard",
        "indexer_L0_index_score",
        "indexer_L0_topk_indices",
        "mla_L0_q",
        "mla_L0_output",
        "layer_L0_output",
        # Layer 1
        "indexer_L1_input_x",
        "mla_L1_output",
        "layer_L1_output",
        # Layer 2
        "layer_L2_output",
        # Final
        "final_logits",
    ]

    results = {
        "first_divergence": None,
        "comparisons": {},
        "missing_official": [],
        "missing_fork": [],
    }

    # Find missing tensors
    all_names = set(official_tensors.keys()) | set(fork_tensors.keys())
    for name in all_names:
        if name not in official_tensors:
            results["missing_official"].append(name)
        if name not in fork_tensors:
            results["missing_fork"].append(name)

    # Compare in order
    first_divergence_found = False
    for name in checkpoint_order:
        if name in official_tensors and name in fork_tensors:
            is_close, stats = compare_tensors(
                name, official_tensors[name], fork_tensors[name]
            )
            results["comparisons"][name] = stats

            if not is_close and not first_divergence_found:
                results["first_divergence"] = name
                first_divergence_found = True

    # Also compare any tensors not in the expected order
    for name in sorted(all_names):
        if name not in results["comparisons"]:
            if name in official_tensors and name in fork_tensors:
                is_close, stats = compare_tensors(
                    name, official_tensors[name], fork_tensors[name]
                )
                results["comparisons"][name] = stats

    return results


def print_report(results: Dict):
    """Print a human-readable report."""
    print("=" * 80)
    print("TENSOR COMPARISON REPORT")
    print("=" * 80)

    # Missing tensors
    if results["missing_official"]:
        print("\n MISSING FROM OFFICIAL:")
        for name in results["missing_official"]:
            print(f"  - {name}")

    if results["missing_fork"]:
        print("\n MISSING FROM HF FORK:")
        for name in results["missing_fork"]:
            print(f"  - {name}")

    # First divergence
    print("\n" + "=" * 80)
    if results["first_divergence"]:
        print(f"FIRST DIVERGENCE: {results['first_divergence']}")
        print("=" * 80)
        print("\nThis is where you should focus debugging!")
        print("The issue is in or before the code that produces this tensor.")
    else:
        print("NO DIVERGENCE FOUND - All tensors match!")
        print("=" * 80)

    # Detailed comparison
    print("\n" + "-" * 80)
    print("DETAILED COMPARISON")
    print("-" * 80)

    for name, stats in results["comparisons"].items():
        if "error" in stats:
            status = f"SHAPE MISMATCH: {stats['official_shape']} vs {stats['fork_shape']}"
            print(f"\n{name}: {status}")
            continue

        if stats["is_close"]:
            status = "MATCH"
        else:
            status = "DIVERGED"

        print(f"\n{name}: {status}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Max abs diff: {stats['max_abs_diff']:.2e}")
        print(f"  Mean abs diff: {stats['mean_abs_diff']:.2e}")
        print(f"  Correlation: {stats['correlation']:.6f}")
        print(f"  Official: mean={stats['official_mean']:.4f}, std={stats['official_std']:.4f}")
        print(f"  Fork:     mean={stats['fork_mean']:.4f}, std={stats['fork_std']:.4f}")

        if not stats["is_close"] and "max_diff_location" in stats:
            print(f"  Max diff at: {stats['max_diff_location']}")
            print(f"    Official value: {stats['official_at_max']:.6f}")
            print(f"    Fork value:     {stats['fork_at_max']:.6f}")


def suggest_fixes(results: Dict):
    """Suggest fixes based on the divergence location."""
    print("\n" + "=" * 80)
    print("SUGGESTED DEBUGGING STEPS")
    print("=" * 80)

    first_div = results.get("first_divergence")
    if not first_div:
        print("\nAll tensors match! The issue might be:")
        print("  1. In the generation loop (not forward pass)")
        print("  2. In KV cache handling")
        print("  3. In sampling/temperature")
        return

    suggestions = {
        "embedding_output": [
            "Check embed_tokens weight loading",
            "Verify vocab size matches",
            "Check input_ids preprocessing",
        ],
        "indexer_L0_input_x": [
            "This is the hidden states entering the indexer",
            "If this diverges, the issue is in embedding or layer norm",
        ],
        "indexer_L0_input_qr": [
            "This is the compressed query (q_a_layernorm output)",
            "Check q_a_proj and q_a_layernorm weights",
        ],
        "indexer_L0_q_after_rope": [
            "Issue is in indexer's RoPE application",
            "Verify using NON-INTERLEAVED RoPE (like Llama)",
            "Check cos/sin slicing to qk_rope_head_dim",
        ],
        "indexer_L0_k_after_rope": [
            "Issue is in indexer's key RoPE",
            "Check k_norm (LayerNorm, not RMSNorm)",
            "Verify wk weight loading",
        ],
        "indexer_L0_q_after_hadamard": [
            "Issue is in Hadamard transform",
            "Verify fast-hadamard-transform is installed",
            "Check scaling: hidden_size ** -0.5",
        ],
        "indexer_L0_index_score": [
            "Issue is in index score computation",
            "Check: scores = ReLU(q @ k.T)",
            "Check: weights = weights_proj(x) * n_heads**-0.5 * softmax_scale",
            "Check: index_scores = (scores * weights).sum(dim=1)",
        ],
        "indexer_L0_topk_indices": [
            "Issue is in top-k selection",
            "Check index_topk config value",
            "Verify attention mask handling",
        ],
        "mla_L0_output": [
            "Issue is in MLA attention computation",
            "Check sparse mask creation",
            "Verify Q/K/V projections match V3",
        ],
        "layer_L0_output": [
            "Issue might be in MoE/MLP",
            "Check router weights",
            "Verify expert computation",
        ],
        "final_logits": [
            "Issue is in lm_head or final norm",
            "Check norm weight loading",
            "Verify lm_head tied weights",
        ],
    }

    if first_div in suggestions:
        print(f"\nDivergence at: {first_div}")
        print("\nSuggested checks:")
        for i, suggestion in enumerate(suggestions[first_div], 1):
            print(f"  {i}. {suggestion}")
    else:
        print(f"\nDivergence at: {first_div}")
        print("No specific suggestions - inspect the code producing this tensor")

    # Check for common patterns
    stats = results["comparisons"].get(first_div, {})
    if stats.get("correlation", 1.0) < 0.5:
        print("\n LOW CORRELATION detected - likely a fundamental mismatch:")
        print("  - Wrong weights loaded")
        print("  - Different algorithm entirely")
        print("  - Data type issue")
    elif stats.get("max_rel_diff", 0) > 0.1:
        print("\n HIGH RELATIVE ERROR detected - likely a scaling issue:")
        print("  - Missing or wrong scaling factor")
        print("  - Different normalization")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=str, default="./saved_tensors/official",
                        help="Directory with official tensors")
    parser.add_argument("--fork", type=str, default="./saved_tensors/hf_fork",
                        help="Directory with HF fork tensors")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance")
    args = parser.parse_args()

    official_dir = Path(args.official)
    fork_dir = Path(args.fork)

    if not official_dir.exists():
        print(f"ERROR: Official tensors not found at {official_dir}")
        print("Run 3_run_official.sh first")
        return

    if not fork_dir.exists():
        print(f"ERROR: HF fork tensors not found at {fork_dir}")
        print("Run 4_run_hf_fork.py first")
        return

    print(f"Loading official tensors from: {official_dir}")
    official_tensors = load_tensors(official_dir)
    print(f"  Found {len(official_tensors)} tensors")

    print(f"Loading HF fork tensors from: {fork_dir}")
    fork_tensors = load_tensors(fork_dir)
    print(f"  Found {len(fork_tensors)} tensors")

    results = analyze_divergence(official_tensors, fork_tensors)

    print_report(results)
    suggest_fixes(results)


if __name__ == "__main__":
    main()
