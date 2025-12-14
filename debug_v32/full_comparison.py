#!/usr/bin/env python3
"""
Full comparison of HF fork against official reference.
Compares: Indexer topk_indices, layer outputs, and generated tokens.
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from collections import defaultdict

RESULTS = {
    "indexer_comparison": [],
    "tensor_comparison": [],
    "generation_comparison": [],
}

CAPTURED_TENSORS = {}
HOOKS = []


def save_tensor(name, tensor):
    if tensor is not None:
        CAPTURED_TENSORS[name] = tensor.detach().cpu().float()


def register_hooks(model, layer_limit=5):
    """Register hooks to capture Indexer outputs and layer outputs."""

    def make_indexer_hook(layer_idx):
        def hook(module, input, output):
            save_tensor(f'layer_{layer_idx}_indexer_topk_indices', output)
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                save_tensor(f'layer_{layer_idx}_attn_output', output[0])
            else:
                save_tensor(f'layer_{layer_idx}_attn_output', output)
        return hook

    for layer_idx in range(min(layer_limit, len(model.model.layers))):
        layer = model.model.layers[layer_idx]

        # Indexer hook
        if hasattr(layer.self_attn, 'indexer') and layer.self_attn.indexer is not None:
            h = layer.self_attn.indexer.register_forward_hook(make_indexer_hook(layer_idx))
            HOOKS.append(h)

        # Attention output hook
        h = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
        HOOKS.append(h)


def remove_hooks():
    for h in HOOKS:
        h.remove()
    HOOKS.clear()


def compare_indexer_indices(hf_indices, ref_dir, layer_idx):
    """Compare Indexer topk_indices against reference."""
    ref_path = os.path.join(ref_dir, f"0000_rank0_layer_{layer_idx}_indexer_topk_indices.pt")
    if not os.path.exists(ref_path):
        return {"status": "ref_not_found", "layer": layer_idx}

    ref_data = torch.load(ref_path)
    ref_indices = ref_data['data'].float()

    result = {
        "layer": layer_idx,
        "hf_shape": list(hf_indices.shape),
        "ref_shape": list(ref_indices.shape),
    }

    if hf_indices.shape != ref_indices.shape:
        result["status"] = "shape_mismatch"
        return result

    # Compare indices
    hf_int = hf_indices.long()
    ref_int = ref_indices.long()

    # Exact match rate
    exact_match = (hf_int == ref_int).float().mean().item()

    # Set match (same set of indices, order may differ)
    total_positions = hf_int.shape[1]
    set_match_count = 0
    for pos in range(total_positions):
        hf_set = set(hf_int[0, pos, :].tolist())
        ref_set = set(ref_int[0, pos, :].tolist())
        if hf_set == ref_set:
            set_match_count += 1
    set_match = set_match_count / total_positions

    result["exact_match_rate"] = exact_match
    result["set_match_rate"] = set_match
    result["hf_first_pos"] = hf_int[0, 0, :min(10, hf_int.shape[-1])].tolist()
    result["ref_first_pos"] = ref_int[0, 0, :min(10, ref_int.shape[-1])].tolist()

    if exact_match > 0.99:
        result["status"] = "exact_match"
    elif set_match > 0.99:
        result["status"] = "set_match"
    elif set_match > 0.9:
        result["status"] = "high_overlap"
    else:
        result["status"] = "diverged"

    return result


def compare_tensor(hf_tensor, ref_dir, tensor_name):
    """Compare a tensor against reference."""
    # Try to load from rank0
    ref_path = os.path.join(ref_dir, f"0000_rank0_{tensor_name}.pt")
    if not os.path.exists(ref_path):
        return {"name": tensor_name, "status": "ref_not_found"}

    ref_data = torch.load(ref_path)
    ref_tensor = ref_data['data'].float()

    result = {
        "name": tensor_name,
        "hf_shape": list(hf_tensor.shape),
        "ref_shape": list(ref_tensor.shape),
    }

    if hf_tensor.shape != ref_tensor.shape:
        result["status"] = "shape_mismatch"
        return result

    # Compute metrics
    diff = (hf_tensor - ref_tensor).abs()
    cos = F.cosine_similarity(hf_tensor.flatten().unsqueeze(0),
                              ref_tensor.flatten().unsqueeze(0)).item()

    result["cosine_similarity"] = cos
    result["max_diff"] = diff.max().item()
    result["mean_diff"] = diff.mean().item()
    result["hf_mean"] = hf_tensor.mean().item()
    result["ref_mean"] = ref_tensor.mean().item()

    if cos > 0.9999:
        result["status"] = "exact_match"
    elif cos > 0.999:
        result["status"] = "very_close"
    elif cos > 0.99:
        result["status"] = "close"
    elif cos > 0.9:
        result["status"] = "similar"
    else:
        result["status"] = "diverged"

    return result


def compare_logits_and_tokens(hf_logits, ref_dir, tokenizer):
    """Compare logits and predicted tokens."""
    # Load and concatenate reference logits from all ranks
    ref_logits_list = []
    for rank in range(8):
        ref_path = os.path.join(ref_dir, f"0000_rank{rank}_logits.pt")
        if os.path.exists(ref_path):
            ref_data = torch.load(ref_path)
            ref_logits_list.append(ref_data['data'].float())

    if not ref_logits_list:
        return {"status": "ref_not_found"}

    ref_logits = torch.cat(ref_logits_list, dim=-1)

    result = {
        "hf_logits_shape": list(hf_logits.shape),
        "ref_logits_shape": list(ref_logits.shape),
    }

    # Get predicted tokens
    hf_pred = hf_logits[0].argmax().item()
    ref_pred = ref_logits[0].argmax().item()

    result["hf_predicted_token_id"] = hf_pred
    result["ref_predicted_token_id"] = ref_pred
    result["hf_predicted_token"] = tokenizer.decode([hf_pred])
    result["ref_predicted_token"] = tokenizer.decode([ref_pred])
    result["tokens_match"] = hf_pred == ref_pred

    # Top-5 comparison
    hf_top5 = hf_logits[0].topk(5).indices.tolist()
    ref_top5 = ref_logits[0].topk(5).indices.tolist()
    result["hf_top5"] = hf_top5
    result["ref_top5"] = ref_top5
    result["top5_overlap"] = len(set(hf_top5) & set(ref_top5))

    # Cosine similarity of logits
    if hf_logits.shape == ref_logits.shape:
        cos = F.cosine_similarity(hf_logits.flatten().unsqueeze(0),
                                  ref_logits.flatten().unsqueeze(0)).item()
        result["logits_cosine_similarity"] = cos

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/models-local/DeepSeek-V3.2-bf16")
    parser.add_argument("--reference-dir", default="/mnt/models-disk/official_tensors")
    parser.add_argument("--prompts-json", default="~/deepseek-v3.2-inference/reference_prompts.json")
    parser.add_argument("--output-file", default="/tmp/comparison_results.json")
    parser.add_argument("--use-sparse", action="store_true", help="Enable sparse attention")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL HF FORK VS OFFICIAL COMPARISON")
    print("=" * 70)

    # Load prompts
    prompts_path = os.path.expanduser(args.prompts_json)
    with open(prompts_path) as f:
        prompts_data = json.load(f)

    prompts = prompts_data["prompts"]
    print(f"Loaded {len(prompts)} prompts")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    config.use_sparse_attention = args.use_sparse

    print(f"\nConfig: use_sparse_attention={config.use_sparse_attention}")

    # Device map
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
    register_hooks(model, layer_limit=5)

    # Process each prompt
    for prompt_info in prompts[:5]:  # First 5 prompts (skip sparse trigger for now)
        prompt_id = prompt_info["id"]
        prompt_name = prompt_info["name"]
        prompt_text = prompt_info["prompt"]
        ref_dir = os.path.join(args.reference_dir, prompt_info["directory"])

        print(f"\n{'='*70}")
        print(f"PROMPT {prompt_id}: {prompt_name}")
        print(f"{'='*70}")
        print(f"Text: {prompt_text[:80]}...")

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt_text}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        device = next(model.parameters()).device

        print(f"Input tokens: {input_ids.shape[1]}")

        # Clear captured tensors
        CAPTURED_TENSORS.clear()

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device))

        logits = outputs.logits[:, -1, :].float().cpu()  # Last position

        # Save embedding output
        # (We'd need to hook the embedding layer for this, skipping for now)

        prompt_results = {
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "input_tokens": input_ids.shape[1],
            "indexer_results": [],
            "tensor_results": [],
            "logits_results": None,
        }

        # Compare Indexer indices
        print("\n--- Indexer Comparison ---")
        for layer_idx in range(5):
            tensor_name = f'layer_{layer_idx}_indexer_topk_indices'
            if tensor_name in CAPTURED_TENSORS:
                result = compare_indexer_indices(CAPTURED_TENSORS[tensor_name], ref_dir, layer_idx)
                prompt_results["indexer_results"].append(result)
                print(f"  Layer {layer_idx}: {result['status']}", end="")
                if "exact_match_rate" in result:
                    print(f" (exact={result['exact_match_rate']:.4f}, set={result['set_match_rate']:.4f})")
                else:
                    print()
            else:
                print(f"  Layer {layer_idx}: NOT CAPTURED")
                prompt_results["indexer_results"].append({"layer": layer_idx, "status": "not_captured"})

        # Compare layer outputs
        print("\n--- Tensor Comparison ---")
        for layer_idx in range(5):
            tensor_name = f'layer_{layer_idx}_attn_output'
            if tensor_name in CAPTURED_TENSORS:
                result = compare_tensor(CAPTURED_TENSORS[tensor_name], ref_dir, tensor_name)
                prompt_results["tensor_results"].append(result)
                print(f"  {tensor_name}: {result['status']}", end="")
                if "cosine_similarity" in result:
                    print(f" (cos={result['cosine_similarity']:.6f})")
                else:
                    print()

        # Compare logits and tokens
        print("\n--- Logits/Token Comparison ---")
        logits_result = compare_logits_and_tokens(logits, ref_dir, tokenizer)
        prompt_results["logits_results"] = logits_result
        print(f"  HF predicted: {logits_result.get('hf_predicted_token', 'N/A')!r} (id={logits_result.get('hf_predicted_token_id', 'N/A')})")
        print(f"  Ref predicted: {logits_result.get('ref_predicted_token', 'N/A')!r} (id={logits_result.get('ref_predicted_token_id', 'N/A')})")
        print(f"  Tokens match: {logits_result.get('tokens_match', 'N/A')}")
        if "logits_cosine_similarity" in logits_result:
            print(f"  Logits cosine: {logits_result['logits_cosine_similarity']:.6f}")

        RESULTS["generation_comparison"].append(prompt_results)

        # Generate output and compare
        print("\n--- Generation Test ---")
        gen_outputs = model.generate(
            input_ids.to(device),
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = gen_outputs[0][input_ids.shape[1]:]  # Only new tokens
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"  Generated ({len(generated_ids)} tokens): {generated_text[:150]}...")
        print(f"  Expected: {prompt_info['expected_output'][:150]}...")

        prompt_results["generated_text"] = generated_text
        prompt_results["generated_token_count"] = len(generated_ids)
        prompt_results["expected_output"] = prompt_info["expected_output"]

    # Remove hooks
    remove_hooks()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(RESULTS, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {args.output_file}")
    print(f"{'='*70}")

    # Print summary
    print("\n=== SUMMARY ===")
    for pr in RESULTS["generation_comparison"]:
        print(f"\nPrompt {pr['prompt_id']} ({pr['prompt_name']}):")

        # Indexer summary
        indexer_statuses = [r.get("status", "unknown") for r in pr["indexer_results"]]
        print(f"  Indexer: {indexer_statuses}")

        # Token match
        if pr["logits_results"]:
            print(f"  Token match: {pr['logits_results'].get('tokens_match', 'N/A')}")
            if "logits_cosine_similarity" in pr["logits_results"]:
                print(f"  Logits cosine: {pr['logits_results']['logits_cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()
