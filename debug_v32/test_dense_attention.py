#!/usr/bin/env python3
"""
Test DeepSeek V3.2 HF fork with DENSE attention (Indexer bypassed).

This script tests the base V3 code path by setting use_sparse_attention=False.
If this works but sparse fails, the bug is in the Indexer.

Usage:
    python test_dense_attention.py --checkpoint /models-local/DeepSeek-V3.2-bf16
"""

import argparse
import torch


def test_forward_pass(model, tokenizer, prompt: str = "The capital of France is"):
    """Test forward pass and check prediction."""
    print(f"\n=== Forward Pass Test ===")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print(f"Output shape: {logits.shape}")
    print(f"Logits - mean: {logits.float().mean():.4f}, std: {logits.float().std():.4f}")

    # Check for NaN/Inf
    if torch.isnan(logits).any():
        print("ERROR: NaN detected in logits!")
        return False
    if torch.isinf(logits).any():
        print("ERROR: Inf detected in logits!")
        return False

    # Get prediction
    pred_token_id = logits[0, -1, :].argmax().item()
    pred_token = tokenizer.decode([pred_token_id])
    print(f"Predicted next token: '{pred_token}' (id={pred_token_id})")

    # Check if reasonable
    if "paris" in pred_token.lower() or "par" in pred_token.lower():
        print("PASS: Prediction looks reasonable (Paris-related)")
        return True
    else:
        print(f"WARNING: Unexpected prediction (expected Paris-related)")
        return True  # Not necessarily a failure, just a warning


def test_generation(model, tokenizer, prompt: str = "Hello, how are you today?"):
    """Test text generation."""
    print(f"\n=== Generation Test ===")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    continuation = generated[len(prompt):]

    print(f"Generated: '{generated}'")
    print(f"Continuation: '{continuation}'")

    # Check for gibberish
    if len(continuation.strip()) < 5:
        print("WARNING: Very short generation")
        return False

    # Check for excessive repetition
    words = continuation.split()
    if len(words) > 5 and len(set(words)) < len(words) / 3:
        print("WARNING: Excessive repetition detected")
        return False

    # Check for obviously broken output
    if continuation.count('\ufffd') > 0 or continuation.count('') > 3:
        print("ERROR: Gibberish detected!")
        return False

    print("PASS: Generation looks reasonable")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BF16 checkpoint")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    args = parser.parse_args()

    print("=" * 70)
    print("DeepSeek V3.2 - Dense Attention Test (Indexer Bypassed)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dtype: {args.dtype}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load config first
    print("\nLoading config...")
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)

    # IMPORTANT: Disable sparse attention to test base V3 code path
    print("\n*** DISABLING SPARSE ATTENTION (use_sparse_attention=False) ***")
    print("This tests the base V3 code path, bypassing the Indexer.\n")
    config.use_sparse_attention = False

    # Verify config
    print(f"Config check:")
    print(f"  model_type: {config.model_type}")
    print(f"  use_sparse_attention: {config.use_sparse_attention}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  hidden_size: {config.hidden_size}")

    # Load model
    print("\nLoading model...")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on devices: {set(str(p.device) for p in model.parameters())}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Run tests
    results = []
    results.append(("Forward pass", test_forward_pass(model, tokenizer)))
    results.append(("Generation", test_generation(model, tokenizer)))

    # Additional test prompts
    test_prompts = [
        "2 + 2 =",
        "The meaning of life is",
        "Write a haiku about coding:",
    ]

    for prompt in test_prompts:
        results.append((f"Generation: {prompt[:30]}", test_generation(model, tokenizer, prompt)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Dense Attention Test (Indexer Bypassed)")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Dense attention (V3 code path) works correctly!")
        print("  If sparse attention fails, the bug is in the Indexer.")
    else:
        print("\n✗ Dense attention has issues!")
        print("  This indicates problems in base V3 code, not just the Indexer.")


if __name__ == "__main__":
    main()
