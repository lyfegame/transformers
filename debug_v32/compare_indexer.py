#!/usr/bin/env python3
"""
Compare HF fork Indexer implementation against official reference.

This script tests the Indexer's topk_indices output to verify the
sparse attention selection mechanism is working correctly.
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


def register_indexer_hooks(model, layer_limit=5):
    """Register forward hooks to capture Indexer intermediate tensors."""

    def make_indexer_pre_hook(layer_idx):
        def hook(module, input):
            if len(input) > 0:
                save_tensor(f'layer_{layer_idx}_indexer_input', input[0])
        return hook

    def make_indexer_hook(layer_idx):
        def hook(module, input, output):
            # Output is just topk_indices tensor (not a tuple)
            save_tensor(f'layer_{layer_idx}_indexer_topk_indices', output)
        return hook

    # Register hooks on Indexer modules
    for layer_idx in range(min(layer_limit, len(model.model.layers))):
        layer = model.model.layers[layer_idx]

        # Check if this layer has an indexer
        if hasattr(layer.self_attn, 'indexer') and layer.self_attn.indexer is not None:
            indexer = layer.self_attn.indexer

            # Pre-hook for input
            h = indexer.register_forward_pre_hook(make_indexer_pre_hook(layer_idx))
            HOOKS.append(h)

            # Post-hook for output
            h = indexer.register_forward_hook(make_indexer_hook(layer_idx))
            HOOKS.append(h)
            print(f'  Registered hooks for layer {layer_idx} indexer')
        else:
            print(f'  Layer {layer_idx}: No indexer found')


def remove_hooks():
    """Remove all registered hooks."""
    for h in HOOKS:
        h.remove()
    HOOKS.clear()


def compare_indexer_outputs(hf_tensors, ref_dir):
    """Compare HF Indexer outputs against reference."""
    print(f"\n{'='*70}")
    print("INDEXER COMPARISON")
    print(f"{'='*70}")

    results = []

    # Compare topk_indices for each layer
    for layer_idx in range(5):
        name = f'layer_{layer_idx}_indexer_topk_indices'

        if name not in hf_tensors:
            print(f"\n--- {name} ---")
            print(f"  NOT CAPTURED in HF fork (indexer may not have run)")
            results.append((name, 'not_captured', float('inf')))
            continue

        hf_indices = hf_tensors[name]

        # Load reference
        ref_path = os.path.join(ref_dir, f"0000_rank0_{name}.pt")
        if not os.path.exists(ref_path):
            print(f"\n--- {name} ---")
            print(f"  Reference not found")
            results.append((name, 'ref_not_found', float('inf')))
            continue

        ref_data = torch.load(ref_path)
        ref_indices = ref_data['data'].float()

        print(f"\n--- {name} ---")
        print(f"  HF shape: {hf_indices.shape}")
        print(f"  Ref shape: {ref_indices.shape}")

        if hf_indices.shape != ref_indices.shape:
            print(f"  SHAPE MISMATCH!")
            results.append((name, 'shape_mismatch', float('inf')))
            continue

        # Compare indices
        # For each position, check if the same set of indices is selected
        hf_int = hf_indices.long()
        ref_int = ref_indices.long()

        # Check exact match
        exact_match = (hf_int == ref_int).float().mean().item()

        # Check if same SET of indices (order may differ)
        set_match_count = 0
        total_positions = hf_int.shape[1]
        for pos in range(total_positions):
            hf_set = set(hf_int[0, pos, :].tolist())
            ref_set = set(ref_int[0, pos, :].tolist())
            if hf_set == ref_set:
                set_match_count += 1
        set_match = set_match_count / total_positions

        # Check top-k overlap (what fraction of ref indices appear in hf)
        overlap_count = 0
        for pos in range(total_positions):
            hf_set = set(hf_int[0, pos, :].tolist())
            ref_set = set(ref_int[0, pos, :].tolist())
            overlap_count += len(hf_set & ref_set)
        avg_overlap = overlap_count / (total_positions * hf_int.shape[-1])

        print(f"  Exact match rate: {exact_match:.4f}")
        print(f"  Set match rate: {set_match:.4f}")
        print(f"  Avg overlap: {avg_overlap:.4f}")

        # Show first position comparison
        print(f"  First position HF:  {hf_int[0, 0, :10].tolist()}")
        print(f"  First position Ref: {ref_int[0, 0, :10].tolist()}")

        # Status
        if exact_match > 0.95:
            status = "EXACT_MATCH"
        elif set_match > 0.95:
            status = "SET_MATCH"
        elif avg_overlap > 0.9:
            status = "HIGH_OVERLAP"
        elif avg_overlap > 0.5:
            status = "PARTIAL_OVERLAP"
        else:
            status = "DIVERGED"

        print(f"  Status: {status}")
        results.append((name, status, 1 - avg_overlap))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/models-local/DeepSeek-V3.2-bf16")
    parser.add_argument("--reference-dir", type=str, default="/mnt/models-disk/official_tensors")
    parser.add_argument("--prompt-id", type=int, default=0, help="0-4 for different prompts")
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
    print("HF Fork Indexer Comparison Test")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt ID: {args.prompt_id}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load config with sparse attention ENABLED
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    config.use_sparse_attention = True  # CRITICAL: Enable sparse to exercise Indexer
    print(f"\nConfig: model_type={config.model_type}")
    print(f"use_sparse_attention={config.use_sparse_attention}")
    print(f"index_topk={config.index_topk}")

    # Build device map
    device_map = {}
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = 7
    device_map['lm_head'] = 7

    for i in range(config.num_hidden_layers):
        if i < 48:
            device_map[f'model.layers.{i}'] = i // 6
        else:
            device_map[f'model.layers.{i}'] = 'cpu'

    print("\nLoading model with sparse attention enabled...")
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

    # Check which layers have indexers
    print("\nChecking for Indexer modules:")
    indexer_layers = []
    for i, layer in enumerate(model.model.layers[:10]):
        has_indexer = hasattr(layer.self_attn, 'indexer') and layer.self_attn.indexer is not None
        if has_indexer:
            indexer_layers.append(i)
            print(f"  Layer {i}: Has indexer")
        else:
            print(f"  Layer {i}: No indexer")

    if not indexer_layers:
        print("\nWARNING: No indexer modules found! Sparse attention may not be implemented.")
        return

    # Register hooks
    print("\nRegistering Indexer hooks:")
    register_indexer_hooks(model, layer_limit=5)

    # Get prompt
    prompt = prompts[args.prompt_id]
    prompt_dir = prompt_dirs[args.prompt_id]
    ref_path = os.path.join(args.reference_dir, prompt_dir)

    print(f"\nPrompt: {prompt[:50]}...")

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = {"input_ids": input_ids.to(device)}

    print(f"Input tokens: {inputs['input_ids'].shape}")
    print(f"Note: seq_len={inputs['input_ids'].shape[1]} vs topk={config.index_topk}")

    # Run forward pass
    CAPTURED_TENSORS.clear()

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Captured {len(CAPTURED_TENSORS)} tensors")
    for name in sorted(CAPTURED_TENSORS.keys()):
        print(f"  {name}: {CAPTURED_TENSORS[name].shape}")

    # Remove hooks
    remove_hooks()

    # Compare with reference
    results = compare_indexer_outputs(CAPTURED_TENSORS, ref_path)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for name, status, diff in results:
        print(f"  {status:20s} {name}")

    # Test generation too
    print(f"\n{'='*70}")
    print("GENERATION TEST")
    print(f"{'='*70}")

    outputs = model.generate(
        input_ids.to(device),
        max_new_tokens=50,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if '<｜Assistant｜>' in response:
        response = response.split('<｜Assistant｜>')[-1]
    print(f"Response: {response[:200]}")


if __name__ == "__main__":
    main()
