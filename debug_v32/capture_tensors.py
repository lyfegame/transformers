#!/usr/bin/env python3
"""
Capture intermediate tensors from official DeepSeek V3.2 inference using PyTorch hooks.

Saves tensors from ALL ranks to compare with HF transformer implementation.
Tensors are saved to persistent disk to survive cluster restart.

Usage:
    cd ~/deepseek-v3.2-inference
    torchrun --nproc-per-node=8 capture_tensors.py \
        --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
        --config config_671B_v3.2.json \
        --output-dir /mnt/models-disk/official_tensors \
        --layer-limit 5
"""

import os
import sys

# CRITICAL: Must import tilelang BEFORE torch for H200 compatibility
# This fixes "tvm.error.InternalError: stod" on H200 GPUs
import tilelang
from tilelang import tvm
_target = tvm.target.Target("cuda")

import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_model

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Transformer, ModelArgs


class TensorCapture:
    """Capture tensors from model forward pass using hooks.

    Saves tensors from ALL ranks, with rank ID in filename.
    This allows comparison with HF transformer implementation.
    """

    def __init__(self, output_dir: str, layer_limit: int = 5, rank: int = 0, world_size: int = 1):
        self.output_dir = output_dir
        self.layer_limit = layer_limit
        self.rank = rank
        self.world_size = world_size
        self.captured: Dict[str, torch.Tensor] = {}
        self.step = 0
        self.hooks = []

        # All ranks create their output directory
        os.makedirs(output_dir, exist_ok=True)

    def _save_tensor(self, name: str, tensor: torch.Tensor):
        """Save tensor with metadata. Saves from ALL ranks with rank ID in filename."""
        # Include rank in filename for all ranks
        filename = f"{self.output_dir}/{self.step:04d}_rank{self.rank}_{name}.pt"

        # Handle different tensor types
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Convert FP8 to float for storage
            data = tensor.detach().cpu().to(torch.float32)
        else:
            data = tensor.detach().cpu().float()

        save_dict = {
            "name": name,
            "rank": self.rank,
            "world_size": self.world_size,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data": data,
            "mean": data.float().mean().item(),
            "std": data.float().std().item() if data.numel() > 1 else 0.0,
            "min": data.float().min().item(),
            "max": data.float().max().item(),
        }
        torch.save(save_dict, filename)

        if self.rank == 0:
            print(f"[Capture] {name}: shape={list(tensor.shape)}, mean={save_dict['mean']:.4f}")

    def _capture_mla_internals(self, module, name_prefix: str, input_tuple, output):
        """Capture MLA internal tensors for DSA debugging."""
        # Input to attention is the hidden state
        if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
            hidden = input_tuple[0]
            self._save_tensor(f"{name_prefix}_input", hidden)

        # Output is the attention output
        self._save_tensor(f"{name_prefix}_output", output)

    def register_hooks(self, model: Transformer):
        """Register forward hooks on key modules for DSA/Indexer debugging."""

        # Hook for embedding output
        def embed_hook(module, input, output):
            self._save_tensor("embedding_output", output)

        self.hooks.append(model.embed.register_forward_hook(embed_hook))

        # Hook for each transformer block
        for i, layer in enumerate(model.layers):
            if self.layer_limit >= 0 and i >= self.layer_limit:
                break

            layer_id = i

            # Hook for attention (MLA) - captures input and output
            def make_attn_hook(lid):
                def hook(module, input, output):
                    # Capture attention input (hidden states)
                    if isinstance(input, tuple) and len(input) > 0:
                        self._save_tensor(f"layer_{lid}_attn_input", input[0])
                    # Capture attention output
                    self._save_tensor(f"layer_{lid}_attn_output", output)
                return hook

            self.hooks.append(layer.attn.register_forward_hook(make_attn_hook(layer_id)))

            # Hook for wq (Q projection) if it exists
            if hasattr(layer.attn, 'wq'):
                def make_wq_hook(lid):
                    def hook(module, input, output):
                        self._save_tensor(f"layer_{lid}_wq_output", output)
                    return hook
                self.hooks.append(layer.attn.wq.register_forward_hook(make_wq_hook(layer_id)))

            # Hook for wkv (compressed KV projection) if it exists
            if hasattr(layer.attn, 'wkv'):
                def make_wkv_hook(lid):
                    def hook(module, input, output):
                        self._save_tensor(f"layer_{lid}_wkv_output", output)
                    return hook
                self.hooks.append(layer.attn.wkv.register_forward_hook(make_wkv_hook(layer_id)))

            # Hook for Indexer if it exists (key for sparse attention debugging)
            if hasattr(layer.attn, 'indexer'):
                def make_indexer_hook(lid):
                    def hook(module, input, output):
                        # Capture indexer input
                        if isinstance(input, tuple) and len(input) > 0:
                            self._save_tensor(f"layer_{lid}_indexer_input", input[0])
                        # output is topk_indices
                        self._save_tensor(f"layer_{lid}_indexer_topk_indices", output)
                    return hook
                self.hooks.append(layer.attn.indexer.register_forward_hook(make_indexer_hook(layer_id)))

                # Hook for indexer's internal projections if they exist
                if hasattr(layer.attn.indexer, 'wq'):
                    def make_idx_wq_hook(lid):
                        def hook(module, input, output):
                            self._save_tensor(f"layer_{lid}_indexer_wq_output", output)
                        return hook
                    self.hooks.append(layer.attn.indexer.wq.register_forward_hook(make_idx_wq_hook(layer_id)))

                if hasattr(layer.attn.indexer, 'wk'):
                    def make_idx_wk_hook(lid):
                        def hook(module, input, output):
                            self._save_tensor(f"layer_{lid}_indexer_wk_output", output)
                        return hook
                    self.hooks.append(layer.attn.indexer.wk.register_forward_hook(make_idx_wk_hook(layer_id)))

            # Hook for wo (output projection) if it exists
            if hasattr(layer.attn, 'wo'):
                def make_wo_hook(lid):
                    def hook(module, input, output):
                        self._save_tensor(f"layer_{lid}_wo_output", output)
                    return hook
                self.hooks.append(layer.attn.wo.register_forward_hook(make_wo_hook(layer_id)))

            # Hook for FFN/MoE
            def make_ffn_hook(lid):
                def hook(module, input, output):
                    # Capture FFN input
                    if isinstance(input, tuple) and len(input) > 0:
                        self._save_tensor(f"layer_{lid}_ffn_input", input[0])
                    # Capture FFN output
                    if isinstance(output, tuple):
                        self._save_tensor(f"layer_{lid}_ffn_output", output[0])
                    else:
                        self._save_tensor(f"layer_{lid}_ffn_output", output)
                return hook

            self.hooks.append(layer.ffn.register_forward_hook(make_ffn_hook(layer_id)))

            # Hook for attention norm if it exists
            if hasattr(layer, 'attn_norm'):
                def make_attn_norm_hook(lid):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            self._save_tensor(f"layer_{lid}_attn_norm_output", output[0])
                        else:
                            self._save_tensor(f"layer_{lid}_attn_norm_output", output)
                    return hook
                self.hooks.append(layer.attn_norm.register_forward_hook(make_attn_norm_hook(layer_id)))

            # Hook for FFN norm if it exists
            if hasattr(layer, 'ffn_norm'):
                def make_ffn_norm_hook(lid):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            self._save_tensor(f"layer_{lid}_ffn_norm_output", output[0])
                        else:
                            self._save_tensor(f"layer_{lid}_ffn_norm_output", output)
                    return hook
                self.hooks.append(layer.ffn_norm.register_forward_hook(make_ffn_norm_hook(layer_id)))

        # Hook for final norm (returns tuple: (output, residual))
        def norm_hook(module, input, output):
            if isinstance(output, tuple):
                self._save_tensor("final_norm_output", output[0])
            else:
                self._save_tensor("final_norm_output", output)

        self.hooks.append(model.norm.register_forward_hook(norm_hook))

        # Hook for LM head
        def head_hook(module, input, output):
            self._save_tensor("logits", output)

        self.hooks.append(model.head.register_forward_hook(head_hook))

        print(f"[Rank {self.rank}] Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def increment_step(self):
        """Increment step counter (call between forward passes)."""
        self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/mnt/models-disk/official_tensors")
    parser.add_argument("--layer-limit", type=int, default=5,
                        help="Number of layers to capture (-1 for all)")
    parser.add_argument("--prompt", type=str, default="What is 2+2?",
                        help="Prompt text (use --prompt-file for long prompts)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to file containing prompt (overrides --prompt)")
    args = parser.parse_args()

    # Read prompt from file if specified
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            args.prompt = f.read().strip()

    # Distributed setup
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    # Load config and model
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))

    if rank == 0:
        print(f"Loading model with config: {model_args}")
        print(f"Capturing first {args.layer_limit} layers from ALL {world_size} ranks")
        print(f"Output directory: {args.output_dir}")

    with torch.device("cuda"):
        model = Transformer(model_args)

    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if rank == 0:
        print("Model loaded")

    # Set up tensor capture - captures from ALL ranks
    capture = TensorCapture(args.output_dir, args.layer_limit, rank, world_size)
    capture.register_hooks(model)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    # Tokenize prompt
    messages = [{"role": "user", "content": args.prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    if rank == 0:
        prompt_preview = args.prompt[:100] + "..." if len(args.prompt) > 100 else args.prompt
        print(f"Prompt: {prompt_preview}")
        print(f"Token count: {len(prompt_tokens)}")
        # Check if sparse attention will be triggered (index_topk is typically 2048)
        index_topk = getattr(model_args, 'index_topk', 2048)
        if len(prompt_tokens) > index_topk:
            print(f"  -> SPARSE ATTENTION TRIGGERED: seq_len ({len(prompt_tokens)}) > index_topk ({index_topk})")
            print(f"     topk_indices will have shape [1, {len(prompt_tokens)}, {index_topk}]")
        else:
            print(f"  -> Dense attention (seq_len ({len(prompt_tokens)}) <= index_topk ({index_topk}))")
            print(f"     topk_indices will have shape [1, {len(prompt_tokens)}, {len(prompt_tokens)}]")

    # Synchronize all ranks before forward pass
    if world_size > 1:
        dist.barrier()

    # Run forward pass (just one step to capture tensors)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")

    with torch.inference_mode():
        logits = model.forward(tokens, start_pos=0)

    # Synchronize all ranks after forward pass
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        # Get top predicted token (note: logits are sharded, so this is partial)
        next_token = logits[0, -1].argmax().item()
        print(f"\nNext token (rank 0 shard): {next_token}")
        print(f"\nTensors saved to: {args.output_dir}")

        # List all captured files
        all_files = sorted(os.listdir(args.output_dir))
        print(f"Total files: {len(all_files)}")

        # Show sample of files
        for f in all_files[:20]:
            print(f"  {f}")
        if len(all_files) > 20:
            print(f"  ... and {len(all_files) - 20} more files")

    # Cleanup
    capture.remove_hooks()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
