#!/usr/bin/env python3
"""
Instrument the official DeepSeek V3.2 model to save intermediate tensors.

Usage:
    1. Copy this script to the cluster
    2. Run: python instrument_official.py
    3. Run inference with DEBUG_SAVE_TENSORS=1:

       DEBUG_SAVE_TENSORS=1 DEBUG_OUTPUT_DIR=/tmp/tensors DEBUG_LAYER_LIMIT=3 \
       torchrun --nproc-per-node=8 generate.py ...

Environment variables:
    DEBUG_SAVE_TENSORS: Set to "1" to enable tensor saving
    DEBUG_OUTPUT_DIR: Directory to save tensors (default: /tmp/debug_tensors)
    DEBUG_LAYER_LIMIT: Number of layers to save (default: 3, set to -1 for all)
"""

import os
import shutil

# Patch to add at the top of model.py (after imports)
DEBUG_HEADER = '''
# ============ DEBUG INSTRUMENTATION ============
import os
_DEBUG_SAVE = os.environ.get("DEBUG_SAVE_TENSORS", "0") == "1"
_DEBUG_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "/tmp/debug_tensors")
_DEBUG_LAYER_LIMIT = int(os.environ.get("DEBUG_LAYER_LIMIT", "3"))
_DEBUG_STEP = [0]  # mutable counter

def _save_tensor(name: str, tensor: torch.Tensor, layer_id: int = -1):
    """Save tensor to file if debugging enabled."""
    if not _DEBUG_SAVE:
        return
    if rank != 0:  # Only save from rank 0
        return
    if layer_id >= 0 and _DEBUG_LAYER_LIMIT >= 0 and layer_id >= _DEBUG_LAYER_LIMIT:
        return

    os.makedirs(_DEBUG_DIR, exist_ok=True)
    step = _DEBUG_STEP[0]
    filename = f"{_DEBUG_DIR}/{step:04d}_{name}.pt"

    # Save tensor info and data
    save_dict = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data": tensor.detach().cpu().float(),  # Convert to float32 for comparison
        "mean": tensor.float().mean().item(),
        "std": tensor.float().std().item(),
        "min": tensor.float().min().item(),
        "max": tensor.float().max().item(),
    }
    torch.save(save_dict, filename)
    print(f"[DEBUG] Saved {name}: shape={list(tensor.shape)}, mean={save_dict['mean']:.6f}, std={save_dict['std']:.6f}")

def _debug_step():
    """Increment debug step counter."""
    _DEBUG_STEP[0] += 1
# ============ END DEBUG INSTRUMENTATION ============
'''

# Patches for Indexer.forward
INDEXER_FORWARD_PATCHES = [
    # After computing q
    ('q = self.wq_b(qr)',
     '''q = self.wq_b(qr)
        _save_tensor(f"indexer_L{self.layer_id}_q_raw", q, self.layer_id)'''),

    # After RoPE on q
    ('q_pe = apply_rotary_emb(q_pe, freqs_cis, False)',
     '''q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        _save_tensor(f"indexer_L{self.layer_id}_q_after_rope", q_pe, self.layer_id)'''),

    # After computing k
    ('k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)',
     '''k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        _save_tensor(f"indexer_L{self.layer_id}_k_after_rope", k_pe, self.layer_id)'''),

    # After Hadamard transform
    ('q = rotate_activation(q)',
     '''q = rotate_activation(q)
        _save_tensor(f"indexer_L{self.layer_id}_q_after_hadamard", q, self.layer_id)'''),

    # After computing index_score
    ('index_score = fp8_index(',
     '''_save_tensor(f"indexer_L{self.layer_id}_weights", weights, self.layer_id)
        index_score = fp8_index('''),

    # After topk selection (need to find the exact line)
    ('topk_indices = index_score.topk',
     '''_save_tensor(f"indexer_L{self.layer_id}_index_score", index_score, self.layer_id)
        topk_indices = index_score.topk'''),
]

# Patches for MLA.forward
MLA_FORWARD_PATCHES = [
    # After computing qr (compressed query)
    ('qr = self.q_norm(self.wq_a(x))',
     '''qr = self.q_norm(self.wq_a(x))
        _save_tensor(f"mla_L{self.layer_id}_qr", qr, self.layer_id)'''),

    # After attention output
    ('self.wo(output)',
     '''_save_tensor(f"mla_L{self.layer_id}_output_pre_wo", output, self.layer_id)
        self.wo(output)'''),
]

# Patches for Block.forward
BLOCK_FORWARD_PATCHES = [
    # Input to block
    ('x = self.attn(x, start_pos, freqs_cis, mask)',
     '''_save_tensor(f"block_L{self.layer_id}_input", x, self.layer_id)
        x = self.attn(x, start_pos, freqs_cis, mask)
        _save_tensor(f"block_L{self.layer_id}_after_attn", x, self.layer_id)'''),

    # After FFN
    ('x = self.ffn(x)',
     '''x = self.ffn(x)
        _save_tensor(f"block_L{self.layer_id}_after_ffn", x, self.layer_id)'''),
]

# Patches for Transformer.forward
TRANSFORMER_FORWARD_PATCHES = [
    # After embedding
    ('h, residual = self.embed(tokens), None',
     '''h, residual = self.embed(tokens), None
        _save_tensor("embedding_output", h)
        _debug_step()'''),

    # Final output before head
    ('self.head(h)',
     '''_save_tensor("final_hidden", h)
        _save_tensor("final_logits", self.head(h))
        self.head(h)'''),
]


def instrument_model(model_path: str, output_path: str):
    """Instrument the model.py file with tensor saving hooks."""

    with open(model_path, 'r') as f:
        content = f.read()

    # Add debug header after imports
    # Find the line after "from kernel import"
    kernel_import_idx = content.find("from kernel import")
    if kernel_import_idx == -1:
        print("ERROR: Could not find 'from kernel import' in model.py")
        return False

    # Find end of that line
    end_of_line = content.find('\n', kernel_import_idx)
    content = content[:end_of_line+1] + DEBUG_HEADER + content[end_of_line+1:]

    # Apply patches (simple string replacement)
    # Note: This is a simple approach - in production, use AST manipulation

    all_patches = (
        INDEXER_FORWARD_PATCHES +
        MLA_FORWARD_PATCHES +
        BLOCK_FORWARD_PATCHES +
        TRANSFORMER_FORWARD_PATCHES
    )

    patches_applied = 0
    for old, new in all_patches:
        if old in content:
            content = content.replace(old, new, 1)  # Replace first occurrence only
            patches_applied += 1
            print(f"Applied patch: {old[:50]}...")
        else:
            print(f"WARNING: Could not find pattern: {old[:50]}...")

    # Write instrumented model
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"\nInstrumented model saved to: {output_path}")
    print(f"Patches applied: {patches_applied}/{len(all_patches)}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Instrument official model for debugging")
    parser.add_argument("--model-path", default="model.py", help="Path to model.py")
    parser.add_argument("--output-path", default="model_instrumented.py", help="Output path")
    parser.add_argument("--backup", action="store_true", help="Create backup of original")
    args = parser.parse_args()

    if args.backup:
        backup_path = args.model_path + ".backup"
        shutil.copy(args.model_path, backup_path)
        print(f"Backup created: {backup_path}")

    success = instrument_model(args.model_path, args.output_path)

    if success:
        print("\nTo use instrumented model:")
        print(f"  1. mv {args.output_path} model.py")
        print("  2. Run with: DEBUG_SAVE_TENSORS=1 DEBUG_OUTPUT_DIR=/tmp/tensors torchrun ...")
        print("  3. Tensors saved to /tmp/tensors/*.pt")


if __name__ == "__main__":
    main()
