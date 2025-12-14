#!/usr/bin/env python3
"""
Test DeepSeek V3.2 HF fork with DENSE attention and CPU offloading.

The BF16 model is 1.3TB but GPUs only have 1.15TB total.
We need to offload some layers to CPU.
"""

import argparse
import torch


def test_forward_pass(model, tokenizer, prompt: str = "The capital of France is"):
    """Test forward pass and check prediction."""
    print(f"\n=== Forward Pass Test ===")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to the model's first device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

    if "paris" in pred_token.lower() or "par" in pred_token.lower():
        print("PASS: Prediction looks reasonable (Paris-related)")
    else:
        print(f"WARNING: Unexpected prediction (expected Paris-related)")
    return True


def test_generation(model, tokenizer, prompt: str = "Hello, how are you today?"):
    """Test text generation."""
    print(f"\n=== Generation Test ===")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

    print("PASS: Generation looks reasonable")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    print("=" * 70)
    print("DeepSeek V3.2 - Dense Attention Test (WITH CPU OFFLOAD)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load config
    print("\nLoading config...")
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)

    # Disable sparse attention
    print("\n*** DISABLING SPARSE ATTENTION ***")
    config.use_sparse_attention = False

    print(f"Config: model_type={config.model_type}, use_sparse_attention={config.use_sparse_attention}")

    # Load model with CPU offloading
    # Set max memory per GPU to leave room for activations
    # 8 GPUs × 140GB = 1.12TB for weights, rest goes to CPU
    print("\nLoading model with CPU offloading...")
    print("Setting max_memory: 140GB per GPU, rest to CPU")

    max_memory = {i: "140GiB" for i in range(8)}
    max_memory["cpu"] = "500GiB"  # Use up to 500GB CPU RAM

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=False,
        offload_folder="/tmp/offload",  # For disk offload if needed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Show device distribution
    device_counts = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        device_counts[device] = device_counts.get(device, 0) + 1

    print(f"\nParameter distribution:")
    for device, count in sorted(device_counts.items()):
        print(f"  {device}: {count} parameters")

    # Run tests
    results = []
    results.append(("Forward pass", test_forward_pass(model, tokenizer)))
    results.append(("Generation", test_generation(model, tokenizer)))

    # Test with reference prompts
    test_prompts = [
        "What is 2+2?",
        "Write a Python function to check if a number is prime.",
    ]

    for prompt in test_prompts:
        results.append((f"Generation: {prompt[:30]}", test_generation(model, tokenizer, prompt)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
