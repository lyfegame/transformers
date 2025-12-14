#!/usr/bin/env python3
"""
Run HuggingFace V3.2 fork with tensor saving for comparison against official.

This script:
1. Loads the model from your fork
2. Hooks into key layers to save intermediate activations
3. Runs the same test prompt as the official code
4. Saves tensors in the same format for comparison

Usage:
    python 4_run_hf_fork.py --checkpoint /path/to/DeepSeek-V3.2-fp8

For multi-GPU:
    accelerate launch --num_processes 8 4_run_hf_fork.py --checkpoint /path/to/ckpt
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add your fork to path (adjust if needed)
FORK_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(FORK_PATH))

OUTPUT_DIR = "./saved_tensors/hf_fork"
LAYER_LIMIT = 3  # Only instrument first N layers


class TensorSaver:
    """Context manager to save tensors during forward pass."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hooks = []
        self.saved = {}

    def save(self, name: str, tensor: torch.Tensor):
        """Save a tensor."""
        if tensor is None:
            return
        tensor = tensor.detach().float().cpu()
        path = self.output_dir / f"{name}.pt"
        torch.save(tensor, path)
        self.saved[name] = tensor.shape
        print(f"[DEBUG] Saved {name}: shape={tuple(tensor.shape)}, "
              f"mean={tensor.mean():.6f}, std={tensor.std():.6f}")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def create_indexer_hook(saver: TensorSaver, layer_idx: int):
    """Create a hook for the indexer module."""

    original_forward = None

    def hook_fn(module, args, kwargs, output):
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        q_compressed = args[1] if len(args) > 1 else kwargs.get('q_compressed')

        # Save inputs
        saver.save(f"indexer_L{layer_idx}_input_x", hidden_states)
        saver.save(f"indexer_L{layer_idx}_input_qr", q_compressed)

        # The output is either topk_indices or (topk_indices, scores)
        if isinstance(output, tuple):
            topk_indices, index_scores = output
            saver.save(f"indexer_L{layer_idx}_index_score", index_scores)
            saver.save(f"indexer_L{layer_idx}_topk_indices", topk_indices)
        else:
            saver.save(f"indexer_L{layer_idx}_topk_indices", output)

        return output

    return hook_fn


def create_attention_hook(saver: TensorSaver, layer_idx: int):
    """Create a hook for the attention module."""

    def hook_fn(module, args, kwargs, output):
        # output is (attn_output, attn_weights, indexer_scores, indexer_kl_target)
        attn_output = output[0]
        saver.save(f"mla_L{layer_idx}_output", attn_output)
        return output

    return hook_fn


def create_layer_hook(saver: TensorSaver, layer_idx: int):
    """Create a hook for the decoder layer."""

    def hook_fn(module, args, kwargs, output):
        # output is (hidden_states, attn_weights, indexer_scores, indexer_kl_target)
        hidden_states = output[0]
        saver.save(f"layer_L{layer_idx}_output", hidden_states)
        return output

    return hook_fn


def instrument_model(model, saver: TensorSaver):
    """Add hooks to save intermediate tensors."""

    # Hook embedding
    def embed_hook(module, input, output):
        saver.save("embedding_output", output)
        return output

    hook = model.model.embed_tokens.register_forward_hook(embed_hook)
    saver.hooks.append(hook)

    # Hook layers
    for i, layer in enumerate(model.model.layers):
        if i >= LAYER_LIMIT:
            break

        # Hook indexer (inside attention)
        if hasattr(layer.self_attn, 'indexer'):
            hook = layer.self_attn.indexer.register_forward_hook(
                create_indexer_hook(saver, i),
                with_kwargs=True
            )
            saver.hooks.append(hook)

        # Hook attention output
        hook = layer.self_attn.register_forward_hook(
            create_attention_hook(saver, i),
            with_kwargs=True
        )
        saver.hooks.append(hook)

        # Hook layer output
        hook = layer.register_forward_hook(
            create_layer_hook(saver, i),
            with_kwargs=True
        )
        saver.hooks.append(hook)

    # Hook final logits
    def lm_head_hook(module, input, output):
        saver.save("final_logits", output)
        return output

    hook = model.lm_head.register_forward_hook(lm_head_hook)
    saver.hooks.append(hook)

    print(f"[DEBUG] Installed {len(saver.hooks)} hooks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DeepSeek V3.2 checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?",
                        help="Test prompt")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save tensors")
    parser.add_argument("--use-sparse", action="store_true",
                        help="Enable sparse attention (default: disabled for debugging)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    print("=== Running HuggingFace V3.2 Fork with Tensor Saving ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use sparse attention: {args.use_sparse}")
    print("")

    # Import from your fork
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load config first to check model type
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=False)
    print(f"Model type: {config.model_type}")
    print(f"use_sparse_attention: {getattr(config, 'use_sparse_attention', 'N/A')}")
    print(f"index_topk: {getattr(config, 'index_topk', 'N/A')}")
    print("")

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=False,  # Use YOUR fork
    )

    # Configure sparse attention
    model.config.use_sparse_attention = args.use_sparse
    model.config.indexer_kl_coef = 0.0  # Disable KL loss for inference

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Check indexer weights
    print("\n=== Checking Indexer Weights ===")
    indexer_params = [(n, p) for n, p in model.named_parameters() if "indexer" in n]
    if not indexer_params:
        print("WARNING: No indexer parameters found!")
    else:
        print(f"Found {len(indexer_params)} indexer parameters")
        for name, param in indexer_params[:4]:  # Show first 4
            print(f"  {name}: shape={tuple(param.shape)}, "
                  f"mean={param.float().mean():.6f}, std={param.float().std():.6f}")
        if len(indexer_params) > 4:
            print(f"  ... and {len(indexer_params) - 4} more")

    # Setup tensor saving
    saver = TensorSaver(args.output_dir)
    instrument_model(model, saver)

    # Tokenize
    print(f"\n=== Running Forward Pass ===")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"Input tokens: {inputs['input_ids'].shape}")

    # Forward pass (no generation, just single forward)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False)

    # Also save logits explicitly
    saver.save("final_logits_explicit", outputs.logits)

    # Cleanup
    saver.cleanup()

    print(f"\n=== Saved {len(saver.saved)} tensors to {args.output_dir} ===")
    for name, shape in sorted(saver.saved.items()):
        print(f"  {name}: {shape}")

    # Also run generation to see output quality
    print(f"\n=== Generation Test ===")
    model.config.use_sparse_attention = args.use_sparse
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,  # Greedy for determinism
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    print(f"Input: {args.prompt}")
    print(f"Output: {generated_text}")

    # Save generation for comparison
    with open(os.path.join(args.output_dir, "generation.txt"), "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Output: {generated_text}\n")
        f.write(f"use_sparse_attention: {args.use_sparse}\n")


if __name__ == "__main__":
    main()
