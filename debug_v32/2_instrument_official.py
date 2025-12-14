#!/usr/bin/env python3
"""
Instrument the official DeepSeek V3.2 model.py to save intermediate tensors.

This script patches model.py to add tensor saving at key checkpoints:
- Embedding output
- Per-layer: Q, K, V, attention output, indexer scores, MoE output
- Final logits

Run this AFTER 1_setup_official.sh
"""

import os
import re

OFFICIAL_CODE_DIR = "./official_inference"
MODEL_PY = os.path.join(OFFICIAL_CODE_DIR, "model.py")
OUTPUT_DIR = "./saved_tensors/official"

# The instrumentation code to inject
INSTRUMENTATION_HEADER = '''
# === DEBUG INSTRUMENTATION (added by debug framework) ===
import os
_DEBUG_SAVE_TENSORS = os.environ.get("DEBUG_SAVE_TENSORS", "0") == "1"
_DEBUG_OUTPUT_DIR = os.environ.get("DEBUG_OUTPUT_DIR", "./saved_tensors/official")
_DEBUG_LAYER_LIMIT = int(os.environ.get("DEBUG_LAYER_LIMIT", "3"))  # Only save first N layers
_DEBUG_TENSORS = {}

def _save_debug_tensor(name: str, tensor, force: bool = False):
    """Save tensor for debugging comparison."""
    if not _DEBUG_SAVE_TENSORS:
        return
    if tensor is None:
        return
    import torch
    # Only save on rank 0 to avoid duplicates
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    os.makedirs(_DEBUG_OUTPUT_DIR, exist_ok=True)
    # Detach and move to CPU
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().float().cpu()
    save_path = os.path.join(_DEBUG_OUTPUT_DIR, f"{name}.pt")
    torch.save(tensor, save_path)
    print(f"[DEBUG] Saved {name}: shape={tuple(tensor.shape)}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
# === END DEBUG INSTRUMENTATION ===
'''

# Patches to apply to specific functions
PATCHES = [
    # Patch 1: Save embedding output in Transformer.forward
    {
        "search": r"(h = self\.embed\(tokens\))",
        "replace": r'''\1
        _save_debug_tensor("embedding_output", h)''',
    },

    # Patch 2: Save indexer inputs/outputs in Indexer.forward
    {
        "search": r"(def forward\(self, x: torch\.Tensor, qr: torch\.Tensor.*?\n)",
        "replace": r'''\1        _save_debug_tensor(f"indexer_L{self.layer_idx}_input_x", x)
        _save_debug_tensor(f"indexer_L{self.layer_idx}_input_qr", qr)
''',
        "in_class": "Indexer"
    },

    # Patch 3: Save Q, K after RoPE in Indexer
    {
        "search": r"(q = torch\.cat\(\[q_pe, q_nope\], dim=-1\))",
        "replace": r'''\1
        _save_debug_tensor(f"indexer_L{self.layer_idx}_q_after_rope", q)''',
        "in_class": "Indexer"
    },
    {
        "search": r"(k = torch\.cat\(\[k_pe, k_nope\], dim=-1\))",
        "replace": r'''\1
        _save_debug_tensor(f"indexer_L{self.layer_idx}_k_after_rope", k)''',
        "in_class": "Indexer"
    },

    # Patch 4: Save after Hadamard transform
    {
        "search": r"(q = rotate_activation\(q\))",
        "replace": r'''\1
        _save_debug_tensor(f"indexer_L{self.layer_idx}_q_after_hadamard", q)''',
        "in_class": "Indexer"
    },
    {
        "search": r"(k = rotate_activation\(k\))",
        "replace": r'''\1
        _save_debug_tensor(f"indexer_L{self.layer_idx}_k_after_hadamard", k)''',
        "in_class": "Indexer"
    },

    # Patch 5: Save index scores and topk
    {
        "search": r"(topk_indices = index_score\.topk.*?\n)",
        "replace": r'''_save_debug_tensor(f"indexer_L{self.layer_idx}_index_score", index_score)
        \1        _save_debug_tensor(f"indexer_L{self.layer_idx}_topk_indices", topk_indices)
''',
        "in_class": "Indexer"
    },

    # Patch 6: Save attention Q, K, V in MLA.forward
    {
        "search": r"(q = self\.wq_b\(qr\).*?\n)",
        "replace": r'''\1        if hasattr(self, 'layer_idx') and self.layer_idx < _DEBUG_LAYER_LIMIT:
            _save_debug_tensor(f"mla_L{self.layer_idx}_q", q)
''',
        "in_class": "MLA"
    },

    # Patch 7: Save attention output
    {
        "search": r"(x = self\.wo\(x\.flatten\(2\)\))",
        "replace": r'''\1
        if hasattr(self, 'layer_idx') and self.layer_idx < _DEBUG_LAYER_LIMIT:
            _save_debug_tensor(f"mla_L{self.layer_idx}_output", x)''',
        "in_class": "MLA"
    },

    # Patch 8: Save layer output in Block.forward
    {
        "search": r"(def forward\(self, x: torch\.Tensor.*?return h)",
        "replace": r'''\1
        if self.layer_idx < _DEBUG_LAYER_LIMIT:
            _save_debug_tensor(f"layer_L{self.layer_idx}_output", h)
        return h''',
        "in_class": "Block",
        "flags": re.DOTALL
    },

    # Patch 9: Save final logits
    {
        "search": r"(logits = self\.head\(h\))",
        "replace": r'''\1
        _save_debug_tensor("final_logits", logits)''',
    },
]


def apply_patches(code: str) -> str:
    """Apply all patches to the model code."""

    # First, add the instrumentation header after imports
    import_end = code.rfind("from ")
    if import_end == -1:
        import_end = 0
    # Find the end of import section
    lines = code.split('\n')
    insert_line = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            insert_line = i + 1

    lines.insert(insert_line, INSTRUMENTATION_HEADER)
    code = '\n'.join(lines)

    # Apply each patch
    for patch in PATCHES:
        search = patch["search"]
        replace = patch["replace"]
        flags = patch.get("flags", 0)

        if "in_class" in patch:
            # For class-specific patches, we need more careful handling
            # For now, just apply globally (the self.layer_idx will scope it)
            pass

        code = re.sub(search, replace, code, flags=flags)

    return code


def main():
    print("=== Instrumenting Official model.py ===")

    # Read original
    with open(MODEL_PY, 'r') as f:
        original_code = f.read()

    # Backup original
    backup_path = MODEL_PY + ".original"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(original_code)
        print(f"Backed up original to {backup_path}")

    # Apply patches
    patched_code = apply_patches(original_code)

    # Write patched version
    with open(MODEL_PY, 'w') as f:
        f.write(patched_code)

    print(f"Patched {MODEL_PY}")
    print("")
    print("To run with tensor saving enabled:")
    print("  export DEBUG_SAVE_TENSORS=1")
    print(f"  export DEBUG_OUTPUT_DIR={OUTPUT_DIR}")
    print("  export DEBUG_LAYER_LIMIT=3  # Only save first 3 layers")
    print("")
    print("Then run: ./3_run_official.sh")


if __name__ == "__main__":
    main()
