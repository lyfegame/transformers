#!/usr/bin/env python3
"""
Quick sanity check for DeepSeek V3.2 HuggingFace fork.

This script performs fast checks WITHOUT needing the official code:
1. Verifies model loads correctly
2. Checks all expected weights are present
3. Tests both dense (V3 fallback) and sparse attention paths
4. Reports any obvious issues

Usage:
    python quick_sanity_check.py --checkpoint /path/to/DeepSeek-V3.2-fp8
"""

import argparse
import sys
from pathlib import Path

# Add fork to path
FORK_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(FORK_PATH))

import torch


def check_config(config):
    """Verify configuration is correct for V3.2."""
    print("\n=== Configuration Check ===")
    issues = []

    # Required V3.2 attributes
    required = {
        "model_type": "deepseek_v32",
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "use_sparse_attention": True,
        "scoring_func": "sigmoid",
    }

    for attr, expected in required.items():
        actual = getattr(config, attr, "MISSING")
        status = "" if actual == expected else ""
        print(f"  {attr}: {actual} (expected: {expected}) {status}")
        if actual != expected and actual != "MISSING":
            issues.append(f"{attr}: got {actual}, expected {expected}")
        elif actual == "MISSING":
            issues.append(f"{attr}: MISSING")

    # Inherited V3 attributes
    inherited = {
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "n_routed_experts": 256,
        "rope_interleave": True,
    }

    print("\n  Inherited from V3:")
    for attr, expected in inherited.items():
        actual = getattr(config, attr, "MISSING")
        status = "" if actual == expected else ""
        print(f"    {attr}: {actual} {status}")

    return issues


def check_weights(model):
    """Verify all expected weights are present and non-zero."""
    print("\n=== Weight Check ===")
    issues = []

    # Check indexer weights exist
    indexer_weights = {}
    for name, param in model.named_parameters():
        if "indexer" in name:
            indexer_weights[name] = param

    print(f"\n  Indexer parameters found: {len(indexer_weights)}")
    expected_per_layer = 4  # wq_b, wk, k_norm.weight, k_norm.bias, weights_proj
    expected_layers = model.config.num_hidden_layers
    expected_total = expected_per_layer * expected_layers

    if len(indexer_weights) == 0:
        issues.append("NO INDEXER WEIGHTS FOUND - checkpoint may be V3 not V3.2")
        print("   NO INDEXER WEIGHTS - this is likely the problem!")
    elif len(indexer_weights) < expected_total:
        print(f"   Found {len(indexer_weights)}, expected ~{expected_total}")
        issues.append(f"Missing indexer weights: {len(indexer_weights)} < {expected_total}")

    # Check first layer's indexer weights
    layer0_indexer = {n: p for n, p in indexer_weights.items() if "layers.0." in n}
    print(f"\n  Layer 0 indexer weights:")
    for name, param in sorted(layer0_indexer.items()):
        mean = param.float().mean().item()
        std = param.float().std().item()
        is_zero = std < 1e-8
        status = " (ZERO!)" if is_zero else ""
        print(f"    {name.split('.')[-1]}: shape={tuple(param.shape)}, "
              f"mean={mean:.4f}, std={std:.4f}{status}")
        if is_zero:
            issues.append(f"Zero weights: {name}")

    # Quick check on core weights
    print("\n  Core model weights (sample):")
    sample_weights = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for name in sample_weights:
        try:
            param = dict(model.named_parameters())[name]
            mean = param.float().mean().item()
            std = param.float().std().item()
            print(f"    {name.split('.')[-1]}: shape={tuple(param.shape)}, std={std:.4f}")
        except KeyError:
            print(f"     MISSING: {name}")
            issues.append(f"Missing weight: {name}")

    return issues


def check_forward_pass(model, tokenizer, use_sparse: bool):
    """Test forward pass with given attention mode."""
    mode = "SPARSE" if use_sparse else "DENSE"
    print(f"\n=== Forward Pass Test ({mode}) ===")
    issues = []

    model.config.use_sparse_attention = use_sparse
    model.config.indexer_kl_coef = 0.0

    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Logits mean: {logits.float().mean().item():.4f}")
        print(f"  Logits std: {logits.float().std().item():.4f}")

        # Check for NaN/Inf
        if torch.isnan(logits).any():
            issues.append(f"{mode}: NaN in logits")
            print(f"   NaN detected in logits!")
        if torch.isinf(logits).any():
            issues.append(f"{mode}: Inf in logits")
            print(f"   Inf detected in logits!")

        # Get predicted token
        pred_token_id = logits[0, -1, :].argmax().item()
        pred_token = tokenizer.decode([pred_token_id])
        print(f"  Next token prediction: '{pred_token}' (id={pred_token_id})")

        # Check if prediction is reasonable
        reasonable_tokens = ["Paris", " Paris", "paris", " paris", "Par", " Par"]
        if pred_token.strip() not in [t.strip() for t in reasonable_tokens]:
            issues.append(f"{mode}: Unexpected prediction '{pred_token}' (expected Paris-related)")
            print(f"   Unexpected prediction - might indicate issues")

    except Exception as e:
        issues.append(f"{mode}: Forward pass failed: {str(e)}")
        print(f"   Forward pass failed: {e}")

    return issues


def check_generation(model, tokenizer, use_sparse: bool):
    """Test text generation."""
    mode = "SPARSE" if use_sparse else "DENSE"
    print(f"\n=== Generation Test ({mode}) ===")
    issues = []

    model.config.use_sparse_attention = use_sparse

    test_prompt = "Hello, how are you"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input: '{test_prompt}'")
        print(f"  Output: '{generated}'")

        # Check for obvious gibberish
        continuation = generated[len(test_prompt):].strip()
        if len(continuation) < 5:
            issues.append(f"{mode}: Generation too short")
        elif continuation.count('') > 3 or continuation.count('\ufffd') > 0:
            issues.append(f"{mode}: Gibberish detected in generation")
            print(f"   Possible gibberish detected!")
        # Check for repetition (common failure mode)
        words = continuation.split()
        if len(words) > 3 and len(set(words)) < len(words) / 2:
            issues.append(f"{mode}: Excessive repetition in generation")
            print(f"   Excessive repetition detected!")

    except Exception as e:
        issues.append(f"{mode}: Generation failed: {str(e)}")
        print(f"   Generation failed: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DeepSeek V3.2 checkpoint")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation tests (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("DeepSeek V3.2 HuggingFace Fork - Quick Sanity Check")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dtype: {args.dtype}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    all_issues = []

    # Load config
    print("\nLoading config...")
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    all_issues.extend(check_config(config))

    # Load model
    print("\nLoading model...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Run checks
    all_issues.extend(check_weights(model))
    all_issues.extend(check_forward_pass(model, tokenizer, use_sparse=False))
    all_issues.extend(check_forward_pass(model, tokenizer, use_sparse=True))

    if not args.skip_generation:
        all_issues.extend(check_generation(model, tokenizer, use_sparse=False))
        all_issues.extend(check_generation(model, tokenizer, use_sparse=True))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not all_issues:
        print("\n All checks passed!")
        print("\nIf you're still seeing issues, run the full comparison:")
        print("  1. ./1_setup_official.sh")
        print("  2. python 2_instrument_official.py")
        print("  3. ./3_run_official.sh")
        print("  4. python 4_run_hf_fork.py --checkpoint ...")
        print("  5. python 5_compare_tensors.py")
    else:
        print(f"\n Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

        print("\n Most likely root causes:")
        if any("INDEXER" in i.upper() or "indexer" in i for i in all_issues):
            print("  - Indexer weights not loaded (checkpoint might be V3 not V3.2)")
            print("  - Config model_type might be wrong")
        if any("SPARSE" in i.upper() for i in all_issues):
            print("  - Issue is specific to sparse attention path")
            print("  - Try debugging with use_sparse_attention=False first")
        if any("DENSE" in i.upper() for i in all_issues):
            print("  - Issue is in base V3 code path")
            print("  - Compare your V3 implementation with official")


if __name__ == "__main__":
    main()
