#!/usr/bin/env python3
"""
Capture tensors from HF fork for comparison with official inference.

This script runs the HF fork and saves intermediate tensors at the same
checkpoints as the official inference to enable layer-by-layer comparison.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from collections import defaultdict

# Storage for captured tensors
CAPTURED_TENSORS = {}
HOOKS = []


def save_tensor(name, tensor):
    """Save tensor to capture dict."""
    if tensor is not None:
        CAPTURED_TENSORS[name] = tensor.detach().cpu().float()


def register_hooks(model):
    """Register forward hooks to capture intermediate tensors."""

    def make_embedding_hook():
        def hook(module, input, output):
            save_tensor('embedding_output', output)
        return hook

    def make_layer_hook(layer_idx):
        def pre_hook(module, input):
            save_tensor(f'layer_{layer_idx}_attn_input', input[0])
        return pre_hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_output, attn_weights, indexer_scores, indexer_kl_target)
            if isinstance(output, tuple):
                save_tensor(f'layer_{layer_idx}_attn_output', output[0])
            else:
                save_tensor(f'layer_{layer_idx}_attn_output', output)
        return hook

    def make_ffn_hook(layer_idx):
        def hook(module, input, output):
            save_tensor(f'layer_{layer_idx}_ffn_output', output)
        return hook

    def make_norm_hook():
        def hook(module, input, output):
            save_tensor('final_norm_output', output)
        return hook

    # Register hooks
    # Embedding
    h = model.model.embed_tokens.register_forward_hook(make_embedding_hook())
    HOOKS.append(h)

    # First 5 layers (to match official capture)
    for layer_idx in range(min(5, len(model.model.layers))):
        layer = model.model.layers[layer_idx]

        # Pre-attention input
        h = layer.register_forward_pre_hook(make_layer_hook(layer_idx))
        HOOKS.append(h)

        # Attention output
        h = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
        HOOKS.append(h)

        # FFN output
        h = layer.mlp.register_forward_hook(make_ffn_hook(layer_idx))
        HOOKS.append(h)

    # Final norm
    h = model.model.norm.register_forward_hook(make_norm_hook())
    HOOKS.append(h)


def remove_hooks():
    """Remove all registered hooks."""
    for h in HOOKS:
        h.remove()
    HOOKS.clear()


def compare_tensors(hf_tensors, ref_dir, prompt_name):
    """Compare HF tensors against reference tensors."""
    print(f"\n{'='*70}")
    print(f"TENSOR COMPARISON: {prompt_name}")
    print(f"{'='*70}")

    results = []

    for name, hf_tensor in hf_tensors.items():
        # Find corresponding reference tensor
        ref_path = os.path.join(ref_dir, f"0000_rank0_{name}.pt")

        if not os.path.exists(ref_path):
            print(f"\n--- {name} ---")
            print(f"  Reference not found at {ref_path}")
            continue

        ref_data = torch.load(ref_path)
        ref_tensor = ref_data['data'].float()

        # Compare shapes
        if hf_tensor.shape != ref_tensor.shape:
            print(f"\n--- {name} ---")
            print(f"  SHAPE MISMATCH: HF={hf_tensor.shape} vs Ref={ref_tensor.shape}")
            results.append((name, 'shape_mismatch', float('inf')))
            continue

        # Compute differences
        abs_diff = (hf_tensor - ref_tensor).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # Relative difference
        ref_abs = ref_tensor.abs()
        rel_diff = (abs_diff / (ref_abs + 1e-8)).mean().item()

        # Cosine similarity
        hf_flat = hf_tensor.flatten()
        ref_flat = ref_tensor.flatten()
        cos_sim = F.cosine_similarity(hf_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()

        print(f"\n--- {name} ---")
        print(f"  Shape: {hf_tensor.shape}")
        print(f"  HF  - mean: {hf_tensor.mean():.6f}, std: {hf_tensor.std():.6f}")
        print(f"  Ref - mean: {ref_tensor.mean():.6f}, std: {ref_tensor.std():.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Rel diff: {rel_diff:.6f}")
        print(f"  Cosine sim: {cos_sim:.6f}")

        # Status
        if max_diff < 1e-3:
            status = "MATCH"
        elif max_diff < 0.1:
            status = "CLOSE"
        elif cos_sim > 0.99:
            status = "SIMILAR (high cosine)"
        else:
            status = "DIVERGED"
        print(f"  Status: {status}")

        results.append((name, status, max_diff))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/models-local/DeepSeek-V3.2-bf16")
    parser.add_argument("--reference-dir", type=str, default="/mnt/models-disk/official_tensors")
    parser.add_argument("--output-dir", type=str, default="/tmp/hf_tensors")
    parser.add_argument("--prompt-id", type=int, default=0, help="0-4 for different prompts")
    parser.add_argument("--use-sparse", action="store_true", help="Enable sparse attention")
    args = parser.parse_args()

    # Prompts matching reference
    prompts = [
        "What is 2+2?",
        "Hello, how are you?",
        "Write a Python function to check if a number is prime.",
        "Explain the theory of relativity in simple terms.",
        "The following is a long document about machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention. Machine learning algorithms are often categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled datasets to train algorithms to classify data or predict outcomes accurately. Unsupervised learning analyzes and clusters unlabeled datasets to discover hidden patterns without human intervention. Reinforcement learning trains algorithms through a system of reward and punishment, learning to take actions that maximize rewards.\n\nQuestion: What are the three main categories of machine learning algorithms?",
    ]

    prompt_dirs = [
        "prompt_0_simple_math",
        "prompt_1_greeting",
        "prompt_2_code_generation",
        "prompt_3_explanation",
        "prompt_4_long_context",
    ]

    print("="*70)
    print("HF Fork Tensor Capture and Comparison")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt ID: {args.prompt_id}")
    print(f"Sparse attention: {args.use_sparse}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load config
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    config.use_sparse_attention = args.use_sparse
    print(f"\nConfig: model_type={config.model_type}, sparse={config.use_sparse_attention}")

    # Build device map (same as working test)
    device_map = {}
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = 7
    device_map['lm_head'] = 7

    for i in range(config.num_hidden_layers):
        if i < 48:
            device_map[f'model.layers.{i}'] = i // 6
        else:
            device_map[f'model.layers.{i}'] = 'cpu'

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=False,
        offload_folder='/tmp/offload',
        offload_state_dict=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")

    # Register hooks
    register_hooks(model)

    # Get prompt
    prompt = prompts[args.prompt_id]
    prompt_dir = prompt_dirs[args.prompt_id]
    ref_path = os.path.join(args.reference_dir, prompt_dir)

    print(f"\nPrompt: {prompt[:50]}...")

    # Tokenize using chat template to match official inference
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = {"input_ids": input_ids.to(device)}

    print(f"Input tokens: {inputs['input_ids'].shape}")

    # Run forward pass (no generation, just one forward)
    CAPTURED_TENSORS.clear()

    with torch.no_grad():
        outputs = model(**inputs)

    # Save logits
    save_tensor('logits', outputs.logits[:, -1, :])  # Last token logits

    print(f"\nCaptured {len(CAPTURED_TENSORS)} tensors")

    # Remove hooks
    remove_hooks()

    # Compare with reference
    results = compare_tensors(CAPTURED_TENSORS, ref_path, prompt_dir)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for name, status, diff in results:
        print(f"  {status:15s} {name}")

    # Save captured tensors
    os.makedirs(args.output_dir, exist_ok=True)
    for name, tensor in CAPTURED_TENSORS.items():
        out_path = os.path.join(args.output_dir, f"{name}.pt")
        torch.save({'name': name, 'data': tensor, 'shape': tensor.shape}, out_path)

    print(f"\nTensors saved to {args.output_dir}")


if __name__ == "__main__":
    main()
