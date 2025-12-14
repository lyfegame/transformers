# DeepSeek V3.2 HuggingFace Fork - Debug Strategy

## Latest Status (2024-12-13)

### ✅ Installation Method (Correct Way)

**Date:** 2024-12-14

**Branch:** `shuyingl/deepseek-v32-minimal-on-v4.57.3`
**Remote:** `lyfegame/transformers`

**Install on cluster:**
```bash
pip install --force-reinstall git+https://github.com/lyfegame/transformers@shuyingl/deepseek-v32-minimal-on-v4.57.3
```

**Verify installation:**
```bash
python3 -c "from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer; print('OK')"
```

**Current status:** Installed via pip on cluster (commit `8cba56f7ce`)

---

## Plan: Make modular_deepseek_v32.py HF-Compliant

**Date:** 2024-12-14

### Current State

The working version (`modeling_deepseek_v32.py` on cluster) diverges from HuggingFace conventions:

| Aspect | Working Version | HF Convention |
|--------|-----------------|---------------|
| RoPE output | `freqs_cis` (complex) | `(cos, sin)` tuple |
| RoPE application | Custom `apply_rotary_emb` | `apply_rotary_pos_emb` from `modeling_rope_utils` |
| Indexer cache | Internal `_k_cache` | HF `Cache` class |
| Source file | Manually edited `modeling_*.py` | Generated from `modular_*.py` |

### Goal

Make `modular_deepseek_v32.py` work with **minimal changes** (Occam's razor).

### Iteration Process

Each iteration:
1. Make ONE minimal change to `modular_deepseek_v32.py`
2. Regenerate `modeling_deepseek_v32.py`
3. Commit and push to branch with clear message
4. Clean pip install on cluster
5. Test ALL 6 prompts
6. Document results below

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Logits cosine similarity | > 0.99 |
| Token prediction | Semantically correct (check output makes sense) |
| Prompt 5 indexer shape | `[1, 2250, 2048]` (true sparse) |
| No errors | All 6 prompts complete without crash |

### Reference Tensors (Verified)

All 6 prompts have reference tensors at `/mnt/models-disk/official_tensors/`:

| Prompt | Directory | Tokens | Files | Indexer Shape |
|--------|-----------|--------|-------|---------------|
| 0 | `prompt_0_simple_math` | 10 | 424 | `[1, 10, 10]` (dense) |
| 1 | `prompt_1_greeting` | 11 | 424 | `[1, 11, 11]` (dense) |
| 2 | `prompt_2_code_generation` | 15 | 424 | `[1, 15, 15]` (dense) |
| 3 | `prompt_3_explanation` | 13 | 424 | `[1, 13, 13]` (dense) |
| 4 | `prompt_4_long_context` | 188 | 424 | `[1, 188, 188]` (dense) |
| 5 | `prompt_5_sparse_trigger` | 2250 | 424 | **`[1, 2250, 2048]` (sparse!)** |

### Iteration Log

#### Iteration 0: Baseline (Current Working Version)

**Date:** 2024-12-14
**Commit:** `8cba56f7ce` - "Use working cluster version with freqs_cis RoPE approach"
**Status:** Working for prompts 0-4, **FAILS for prompt 5 (sparse)**

Results from `/tmp/comparison.log` and `/tmp/prompt5_test.log`:

| Prompt | Tokens | Token Match | Logits Cosine | Output Semantic |
|--------|--------|-------------|---------------|-----------------|
| 0 | 10 | ✅ "2" | 0.999 | Correct |
| 1 | 11 | ✅ "Hello" | 0.999 | Correct |
| 2 | 15 | ✅ "Here" | 0.998 | Correct |
| 3 | 12 | ✅ "Of" | 0.994 | Correct |
| 4 | 188 | ✅ "Based" | 0.999 | Correct |
| 5 | 2250 | ❌ HF="Based" Ref="**" | 0.989 | **MISMATCH** |

**Prompt 5 Detailed Results (True Sparse Attention):**

| Layer | Indexer Shape | Jaccard | Overlap |
|-------|---------------|---------|---------|
| 0 | [1, 2250, 2048] ✅ | 0.995 | 2043/2048 |
| 1 | [1, 2250, 2048] ✅ | 0.988 | 2036/2048 |
| 2 | [1, 2250, 2048] ✅ | 0.993 | 2041/2048 |
| 3 | [1, 2250, 2048] ✅ | 0.992 | 2040/2048 |
| 4 | [1, 2250, 2048] ✅ | 0.986 | 2033/2048 |

**Analysis:**
- Indexer is correctly selecting 2048 of 2250 tokens (true sparse)
- Indexer overlap with reference is ~99% (very high)
- But final logits diverge (cosine 0.989 < 0.99 threshold)
- Token prediction differs: HF="Based", Ref="**"

**Hypothesis:** The issue is NOT in the Indexer's token selection, but in how the sparse mask is applied to attention or subsequent computation.

**Next:** Investigate attention computation with sparse mask

#### Iteration 1: Regenerated from Modular (Baseline)

**Date:** 2024-12-14
**Commit:** `240901cc9c` - "Iteration 1: Regenerate modeling from modular for baseline test"
**Status:** ❌ FAILED - NameError crash

**Changes from Iteration 0:**
- Regenerated `modeling_deepseek_v32.py` from `modular_deepseek_v32.py`
- Uses HF-style `(cos, sin)` tuple instead of `freqs_cis` (complex)
- Uses `apply_rotary_pos_emb` instead of custom `apply_rotary_emb`

**Results:**
| Prompt | Token Match | Logits Cosine | Error |
|--------|-------------|---------------|-------|
| 0 | ❌ | 0.931 | Tokens mismatch |
| 1-5 | - | - | Crash during generation |

**Error:**
```
NameError: name 'DeepseekV3Attention' is not defined. Did you mean: 'DeepseekV32Attention'?
  File "modeling_deepseek_v32.py", line 620
    attn_output, attn_weights = DeepseekV3Attention.forward(
```

**Root Cause:** The modular file calls `DeepseekV3Attention.forward(self, ...)` which works in modular context (class is imported), but breaks in generated standalone file where `DeepseekV3Attention` is not defined.

---

#### Fix 1: Replace DeepseekV3Attention.forward with super().forward

**Date:** 2024-12-14
**Commit:** `2832e50657` - "Fix 1: Replace DeepseekV3Attention.forward with super().forward"
**Status:** ❌ FAILED - TypeError

**Change Made (modular_deepseek_v32.py line 686):**
```python
# BEFORE:
DeepseekV3Attention.forward(self, ...)
# AFTER:
super().forward(...)
```

**Results:**
- First forward pass succeeded but logits diverged (cosine 0.931)
- Generation crashed:
```
TypeError: _forward_unimplemented() got an unexpected keyword argument 'hidden_states'
```

**Root Cause:** Generated standalone has `class DeepseekV32Attention(nn.Module)`, not from `DeepseekV3Attention`. So `super().forward()` → `nn.Module.forward()` which is unimplemented.

---

#### Fix 2: Use _forward_dense_warmup instead of super().forward

**Date:** 2024-12-14
**Commit:** `c90e8d9a10` - "Fix 2: Use _forward_dense_warmup instead of super().forward"
**Status:** ⚠️ PARTIAL SUCCESS - No crash but logits diverge

**Change Made:** Replace `super().forward(...)` with `self._forward_dense_warmup(..., output_indexer_scores=False, output_indexer_kl_target=False, ...)`

**Results:**
| Prompt | Token Match | Logits Cosine | Indexer |
|--------|-------------|---------------|---------|
| 0 (simple_math) | ❌ | 0.931 | ✅ set_match |
| 1 (greeting) | ✅ | 0.929 | ✅ set_match |
| 2 (code_generation) | ❌ | 0.884 | ✅ set_match |
| 3 (explanation) | ✅ | 0.919 | ✅ set_match |
| 4 (long_context) | ❌ | 0.967 | ✅ set_match |

**Layer-by-Layer Divergence (Prompt 4):**
```
layer_0: cos=0.997 (close)
layer_1: cos=0.912 (diverging)
layer_2: cos=0.766 (diverged)
layer_3: cos=0.879 (diverged)
layer_4: cos=0.734 (diverged)
```

**Analysis:**
1. ✅ NO CRASH - `_forward_dense_warmup` works as fallback
2. ✅ INDEXER WORKS - All "set_match" on all prompts
3. ❌ LOGITS DIVERGE - All below 0.93, not meeting 0.99 threshold
4. Divergence starts at layer 1, suggesting core attention issue

**Root Cause:** The RoPE implementation or MLA computation differs from reference. The modular uses HF's `apply_rotary_pos_emb`/`apply_rotary_pos_emb_interleave` while the working cluster version used custom `apply_rotary_emb(x, freqs_cis, interleaved=True)` with complex numbers.

---

### ⚠️ Partial Verification - Gaps Remain

**Date:** 2024-12-13

**Summary:** Single forward pass tensors show >99% cosine similarity, but full verification incomplete.

### Comparison Results (Partial Run)

**Date:** 2024-12-13

Results from `full_comparison.py` with `--use-sparse` enabled (partial - output truncated):

| Prompt | Tokens | Indexer Set Match | Token Match | HF Predicted | Ref Predicted |
|--------|--------|-------------------|-------------|--------------|---------------|
| 0 (math) | 10 | 100% | ✅ Yes | "2" | "2" |
| 1 (greeting) | 11 | 100% | ✅ Yes | "Hello" | "Hello" |
| 2 (code) | 15 | 100% | ✅ Yes | "Here" | "Here" |
| 3 (explain) | ? | ? | ? | ? | ? |
| 4 (long ctx) | 188 | ? | ? | ? | ? |
| 5 (sparse) | **2250** | **NOT TESTED** | ? | ? | ? |

**Note:** For prompts 0-4, `seq_len < index_topk (2048)`, so Indexer returns ALL indices (dense behavior). The 100% set match is expected since both return all tokens.

**Prompt 5 is the only true test of sparse attention and has NOT been tested with the HF fork.**

#### What Was Verified ✅

1. **Single forward pass tensor comparison** (prompt_0 only):

| Tensor | Cosine Similarity | Notes |
|--------|-------------------|-------|
| `embedding_output` | 0.999998 | Perfect match |
| `layer_0_attn_output` | 0.999868 | Close |
| `layer_0_ffn_output` | 0.999836 | Close |
| `layer_1_attn_output` | 0.999794 | Close |
| `layer_1_ffn_output` | 0.999228 | Close |
| `final_norm_output` | 0.997869 | Close |
| `logits` | 0.999005 | Close |

2. **First token prediction matches**: Token 20 ("2") for "What is 2+2?"

3. **Generation produces coherent output** (but not compared to reference)

#### What Was NOT Verified ❌

1. **Indexer topk_indices** - NOT compared against reference
   - Hooks captured inputs but OOM prevented full comparison
   - Critical for sparse attention correctness

2. **Multi-token generation** - NOT compared token-by-token against reference
   - Only verified first token matches
   - Small errors compound over autoregressive generation

3. **All 5 reference prompts** - Only tested prompt_0 thoroughly
   - prompt_1_greeting: Not fully tested
   - prompt_2_code_generation: Not tested
   - prompt_3_explanation: Not tested
   - prompt_4_long_context: Not tested

4. **Generation length** - Did NOT generate matching length to reference
   - Reference outputs have specific lengths
   - Need to generate at least as many tokens and compare

#### Why 99% Cosine Similarity May Not Be Enough

- Small logit differences can flip token predictions
- Errors compound over autoregressive generation
- The TRUE test: **generate same tokens as reference**

#### Remaining Work

1. Clear GPU memory (official inference running)
2. Generate all 5 prompts with length matching reference
3. Compare token-by-token with reference
4. Compare Indexer topk_indices against reference
5. Verify sparse attention path exercised correctly

---

### Previous Dense Attention Test

**Date:** 2024-12-14 05:00 UTC

The HuggingFace fork successfully generates coherent output with BF16 checkpoint:

```
Prompt: "The capital of France is"
Output: The capital of France is Paris. 法国的首都是巴黎。
        The capital of Italy is Rome. 意大利的首都是罗马。
        The capital of Spain is Madrid. 西班牙的首都是马德里。
        The capital of Portugal is Lisbon. 葡萄牙的首都是里斯本。
```

### Summary of Fixes Applied

| Issue | Root Cause | Fix |
|-------|------------|-----|
| `weight_scale_inv on meta device` | BF16 checkpoint still had `quantization_config` with `quant_method: "fp8"` | Removed `quantization_config` from config.json |
| `'DeepseekV32Config' has no attribute 'mlp_bias'` | Missing config attribute | Added `mlp_bias: false` to config.json |
| `RecursionError: DeepseekV3Attention.forward` | Bug in installed model file calling undefined class | Copied correct `modeling_deepseek_v32.py` from local codebase |
| OOM at 35/163 shards | Model (1.3TB) > GPU capacity (1.15TB) | Custom device map: 48 layers on GPUs, 13 on CPU |

### Test Configuration That Works

```python
# Custom device map: spread 48 layers across 8 GPUs, 13 layers on CPU
device_map = {}
device_map['model.embed_tokens'] = 0
device_map['model.norm'] = 7
device_map['lm_head'] = 7

for i in range(61):  # 61 layers total
    if i < 48:
        device_map[f'model.layers.{i}'] = i // 6  # 6 layers per GPU
    else:
        device_map[f'model.layers.{i}'] = 'cpu'

# Load with custom device map
model = AutoModelForCausalLM.from_pretrained(
    '/models-local/DeepSeek-V3.2-bf16',
    config=config,  # config.use_sparse_attention = False
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=False,
    offload_folder='/tmp/offload',
    offload_state_dict=True,
)
```

### Required Config Changes for BF16 Checkpoint

The BF16 checkpoint's `config.json` needs these changes:

```bash
# On cluster:
python3 << 'EOF'
import json
with open('/models-local/DeepSeek-V3.2-bf16/config.json', 'r') as f:
    config = json.load(f)

# Remove FP8 quantization config (prevents FP8Linear usage)
if 'quantization_config' in config:
    del config['quantization_config']

# Add missing attributes
config['mlp_bias'] = False
config['use_sparse_attention'] = False  # For dense attention test

with open('/models-local/DeepSeek-V3.2-bf16/config.json', 'w') as f:
    json.dump(config, f, indent=4)
EOF
```

### Files to Copy to Cluster

The cluster's installed transformers has bugs. Copy these files from local:

```bash
# From local machine:
gcloud compute scp /path/to/transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py \
    h200-mig-cluster-rn1h:~/.local/lib/python3.10/site-packages/transformers/models/deepseek_v32/ \
    --zone=europe-west1-b --project=fundamental-labs --tunnel-through-iap

gcloud compute scp /path/to/transformers/src/transformers/models/deepseek_v32/configuration_deepseek_v32.py \
    h200-mig-cluster-rn1h:~/.local/lib/python3.10/site-packages/transformers/models/deepseek_v32/ \
    --zone=europe-west1-b --project=fundamental-labs --tunnel-through-iap
```

### Next Steps

1. **Compare HF fork output vs official reference outputs** - Verify quality matches
2. **Test sparse attention** - Re-enable `use_sparse_attention=True` and test Lightning Indexer
3. **Fix the git repo** - Push the bug fixes to the transformers fork

---

## Quick Start for New Agents

**Goal:** Debug HF transformers fork that generates gibberish. Official inference works correctly.

### What's Ready

| Resource | Location | Description |
|----------|----------|-------------|
| **Reference tensors** | `/mnt/models-disk/official_tensors/` on cluster | 2,544 files from official model (6 prompts × 424 files) |
| **Reference prompts** | `debug_v32/reference_prompts.json` (local) | 6 prompts with expected outputs |
| **Long prompt file** | `debug_v32/long_prompt_sparse.txt` (local) | 2250-token prompt that triggers sparse attention |
| **Capture scripts** | `debug_v32/capture_tensors.py`, `capture_all_prompts.sh` | Re-run tensor capture if needed |
| **BF16 checkpoint** | `/models-local/DeepSeek-V3.2-bf16` on cluster | For HF fork testing (1.3TB) |
| **Official FP8 checkpoint** | `/models-local/DeepSeek-V3.2-converted-mp8` on cluster | Working with tilelang fix |

### Next Task: Compare Weights

1. **Install HF fork on cluster:**
   ```bash
   pip install git+https://github.com/huggingface/transformers@shuyingl/deepseek-v32-minimal-on-v4.57.3
   ```

2. **Run HF fork and capture tensors** for same prompts

3. **Compare tensors** to find divergence point:
   - Use `debug_v32/compare_tensors.py` or write custom comparison
   - Key tensors: `embedding_output`, `layer_N_indexer_topk_indices`, `layer_N_attn_output`

### Cluster Access

```bash
# SSH to cluster
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --tunnel-through-iap

# Key paths on cluster
/mnt/models-disk/official_tensors/     # Captured reference tensors
/models-local/DeepSeek-V3.2-bf16/      # BF16 checkpoint for HF fork
/models-local/DeepSeek-V3.2-fp8/       # Original FP8 checkpoint
~/deepseek-v3.2-inference/             # Official inference code
```

---

## Agent Coordination Status

**Last Updated:** 2024-12-14 05:00 UTC

| Task | Status | Agent | Notes |
|------|--------|-------|-------|
| Checkpoint conversion MP=8 | ✅ Done | - | `/models-local/DeepSeek-V3.2-converted-mp8` (8 shards) |
| Checkpoint conversion MP=16 | ✅ Done | - | `/models-local/DeepSeek-V3.2-converted` |
| **FP8 → BF16 conversion** | ✅ Done | - | 163/163 shards, 1.3TB, ~31 min |
| HF fork installed | ✅ Done | - | Manually copied model files to rn1h |
| **Official FP8 inference** | ✅ WORKING | - | tilelang fix applied! See below |
| **Tensor capture** | ✅ Done | - | 5 prompts × 424 files = 2,120 files, 2.2GB on persistent disk |
| HF fork sanity check | ⏳ Skipped | - | Used dense attention test instead |
| **Dense attention test** | ✅ WORKING | - | Coherent output! See top of README |
| **Sparse attention test** | ⚠️ PARTIAL | - | Generates output but Indexer indices NOT compared |
| **Compare HF vs official** | ⚠️ PARTIAL | - | Single forward pass only, multi-token NOT verified |
| Compare Indexer topk_indices | ❌ NOT DONE | - | OOM prevented comparison |
| Generate all 5 prompts | ❌ NOT DONE | - | Only prompt_0 tested |
| Token-by-token comparison | ❌ NOT DONE | - | Need to match reference generation length |

**✅ DATA SURVIVES:** `/models-local/` → `/mnt/models-disk` (persistent disk, not NFS!). Cluster resize killed the conversion process but data on disk survives.

### Checkpoints Available

| Path | Format | Size | Status |
|------|--------|------|--------|
| `/models-local/DeepSeek-V3.2-fp8` | HF FP8 | 643G | ✅ On persistent disk |
| `/models-local/DeepSeek-V3.2-bf16` | HF BF16 | 1.3T | ✅ Complete (163 shards) |
| `/models-local/DeepSeek-V3.2-converted-mp8` | Official FP8 MP=8 | 641G | ✅ Complete (8 shards) |
| `/models-local/DeepSeek-V3.2-converted` | Official FP8 MP=16 | - | ✅ Complete |
| `/models-local/DeepSeek-V3-bf16` | HF BF16 | ~1.3T | ✅ V3 reference |

**Storage architecture:**
- `/models-local/` → symlink to `/mnt/models-disk` (persistent disk, fast)
- `/models/` → symlink to `/mnt/gcs/models` (GCS mount, slow)

### Known Issues

1. **FP8 Loading with accelerate fails**: The `quick_sanity_check.py` script fails with:
   ```
   ValueError: weight_scale_inv is on the meta device, we need a `value` to put in on 0.
   ```
   This happens because `accelerate`'s `device_map="auto"` doesn't properly handle FP8 scale tensors.

   **Solution:** Converting FP8 → BF16 checkpoint (in progress). The BF16 checkpoint will work with standard HF loading.

2. **Unused weights warning for layer 61**: The checkpoint has extra weights in `model.layers.61` (MoE head). This is expected and can be ignored.

3. **tilelang + PyTorch CUDA initialization conflict** ✅ FIXED:
   - **Symptom**: `tvm.error.InternalError: stod` when creating TVM Target
   - **Root cause**: When PyTorch is imported BEFORE tilelang and initializes CUDA, TVM's target detection fails (std::stod parsing error)
   - **Fix discovered**: Import tilelang at the VERY TOP of generate.py AND kernel.py, BEFORE any torch imports
   - **Status**: ✅ WORKING on H200 with torchrun!

   **The fix (apply to both generate.py and kernel.py):**
   ```python
   # CRITICAL: Import tilelang BEFORE torch to fix TVM on H200
   import tilelang
   from tilelang import tvm
   _target_check = tvm.target.Target("cuda")  # Force TVM init early
   import sys
   print(f'[kernel.py] TVM initialized early: {_target_check}', file=sys.stderr)

   import torch  # Now safe to import torch
   # ... rest of imports ...
   ```

   **Verified working output:**
   ```
   [kernel.py] TVM initialized early: cuda -keys=cuda,gpu -arch=sm_90 -max_num_threads=1024 -thread_warp_size=32
   ModelArgs(...)
   load model
   I'm DeepSeek 👋
   NCCL version 2.21.5+cuda12.4
   Prompt: What is 2+2?
   Completion: That's a classic! **2 + 2 = 4**
   ```

   **Files to patch:**
   - `~/deepseek-v3.2-inference/generate.py` - add tilelang import at line 1
   - `~/deepseek-v3.2-inference/kernel.py` - add tilelang import at line 1

4. **Chat template missing**: The V3.2 tokenizer doesn't include a chat_template. Copy from V3:
   ```bash
   cp /models-local/DeepSeek-V3-bf16/tokenizer_config.json /models-local/DeepSeek-V3.2-converted-mp8/
   ```

---

## GPU Usage & Cluster Status

**Current state (01:05 UTC):** 1 node remaining after resize

| Node | Name | GPUs | Status |
|------|------|------|--------|
| 0 | h200-mig-cluster-rn1h | 8 | 🟢 Available |

**Storage:** `/models-local/` → persistent disk (survives resize!)

**What happened during resize:**
- MIG deleted 4 nodes including m019 (was running conversion)
- Conversion process killed at 71% (116/163 shards)
- Data on persistent disk `/mnt/models-disk` survives
- MP=8 and MP=16 conversions are complete

**Resize commands:**
```bash
# Current size: 1
gcloud compute instance-groups managed resize h200-mig-cluster --size=1 --zone=europe-west1-b --project=fundamental-labs

# Scale up if needed:
gcloud compute instance-groups managed resize h200-mig-cluster --size=2 --zone=europe-west1-b --project=fundamental-labs
```

---

## Dependency Installation Guide

### Required Packages for Official Inference

```bash
# On cluster (gcloud ssh first)
pip install torch transformers safetensors

# tilelang - USE 0.1.6, NOT 0.1.7 (has kernel bug)
pip install tilelang==0.1.6

# jinja2 (needs >=3.1.0 for chat templates)
pip install --upgrade jinja2

# fast-hadamard-transform - NO PREBUILT WHEEL for cu124/torch2.6!
# Must build from source:
cd /tmp
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install . --no-build-isolation
```

### Why fast-hadamard-transform build is needed:
- Prebuilt wheels only exist for cu118/cu122 + torch ≤2.4
- Our cluster has cu124 + torch 2.6
- Building from source takes ~3 minutes, requires ninja

### Verify Installation:
```bash
python3 -c "import tilelang; print(f'tilelang: {tilelang.__version__}')"
python3 -c "from fast_hadamard_transform import hadamard_transform; print('fast_hadamard_transform: OK')"
python3 -c "import jinja2; print(f'jinja2: {jinja2.__version__}')"
```

---

## V3.2 Official Inference Path

### Overview

To get ground truth outputs from V3.2 for debugging the HF fork, we need to run the official inference code. There are two paths:

| Path | Checkpoint | Inference | Status |
|------|------------|-----------|--------|
| **FP8** | HF FP8 → Official FP8 (MP=8) | Uses tilelang kernels | ✅ WORKING (tilelang import fix applied) |
| **BF16** | HF FP8 → HF BF16 → Official BF16 (MP=8) | Standard PyTorch | ⏳ Optional (conversion 71%) |

### FP8 Path (Recommended - Working!)

**Location:** `~/deepseek-v3.2-inference/` on rn1h

**Run inference:**
```bash
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --project=fundamental-labs -- "
cd ~/deepseek-v3.2-inference
unset NCCL_SOCKET_IFNAME  # Required for single-node
nohup torchrun --nproc-per-node=8 generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
    --config config_671B_v3.2.json \
    --input-file test_prompts.txt \
    --max-new-tokens 100 \
    --temperature 0.6 \
    > /tmp/inference.log 2>&1 &
"
# Check results:
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --project=fundamental-labs -- "tail -50 /tmp/inference.log"
```

### Reference Prompts & Expected Outputs (Ground Truth)

**Location:** `~/deepseek-v3.2-inference/reference_outputs/official_outputs.log` on rn1h

These are the reference outputs from official V3.2 FP8 inference. Use these to compare against HF fork outputs.

**Settings:** `temperature=0.0`, `max_new_tokens=200`

---

#### Prompt 1: Simple Math
```
Prompt: What is 2+2?
Completion: 2 + 2 = 4
```

---

#### Prompt 2: Conversation
```
Prompt: Hello, how are you?
Completion: Hello! I'm doing well, thank you for asking! 😊 How are you today? Is there anything I can help you with?
```

---

#### Prompt 3: Code Generation
```
Prompt: Write a Python function to check if a number is prime.
Completion: Here's a Python function to check if a number is prime:

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

---

#### Prompt 4: Explanation
```
Prompt: Explain the theory of relativity in simple terms.
Completion: Of course! Here is the theory of relativity explained in simple terms...

### The Core Idea
At its heart, **relativity is about understanding how space, time, gravity, and motion are all connected.**

### Part 1: Special Relativity (1905)
This deals with things moving at constant, very high speeds (close to the speed of light).

**Key Ideas:**
1. **The Speed of Light is the Universe's Speed Limit:** Nothing can travel faster than light...
```

---

#### Prompt 5: Long Context (Tests Sparse Attention)
```
Prompt: The following is a long document about machine learning. [~200 tokens of context]... Question: What are the three main categories of machine learning algorithms?

Completion: Based on the document provided, the three main categories of machine learning algorithms are:
1. **Supervised learning**
2. **Unsupervised learning**
3. **Reinforcement learning**
```

---

---

## Layer-by-Layer Tensor Capture (Ground Truth)

**Location:** `/mnt/models-disk/official_tensors/` on rn1h (persistent disk - survives cluster restart!)

**Summary:**
- **2,544 tensor files** total (424 files × 6 prompts)
- **~2.5GB total** on persistent disk
- **Layers captured:** 0-4 (first 5 layers, all include indexer)
- **All 8 ranks** saved per tensor

### Reference Prompts Captured

All 6 reference prompts captured to **separate directories**:

| Directory | Prompt | Tokens | Sparse Triggered? | Expected Output |
|-----------|--------|--------|-------------------|-----------------|
| `prompt_0_simple_math/` | "What is 2+2?" | 10 | No (10 < 2048) | "2 + 2 = 4" |
| `prompt_1_greeting/` | "Hello, how are you?" | 11 | No (11 < 2048) | "Hello! I'm doing well..." |
| `prompt_2_code_generation/` | "Write a Python function..." | 15 | No (15 < 2048) | `def is_prime(n):...` |
| `prompt_3_explanation/` | "Explain relativity..." | 13 | No (13 < 2048) | "At its heart, relativity..." |
| `prompt_4_long_context/` | ML document + question | 188 | No (188 < 2048) | "1. Supervised 2. Unsupervised 3. Reinforcement" |
| `prompt_5_sparse_trigger/` | Transformer architecture doc | **2250** | **YES (2250 > 2048)** | "MLA, Lightning Indexer, RoPE..." |

**IMPORTANT:** Only `prompt_5_sparse_trigger` actually triggers sparse attention! The indexer's `index_topk=2048`, so only when `seq_len > 2048` does the indexer select a true subset of tokens.

**Local reference file:** `debug_v32/reference_prompts.json` - Contains all prompts and expected outputs
**Long prompt file:** `debug_v32/long_prompt_sparse.txt` - The 2250-token prompt for sparse testing

### Captured Tensors Per Layer

For each layer (0-4), the following tensors are captured from ALL 8 ranks:

| Tensor | Shape (prompt_0) | Shape (prompt_5) | Description |
|--------|------------------|------------------|-------------|
| `embedding_output` | [1, 10, 7168] | [1, 2250, 7168] | Initial embedding output |
| `layer_N_attn_norm_output` | [1, 10, 7168] | [1, 2250, 7168] | Pre-attention layer norm |
| `layer_N_attn_input` | [1, 10, 7168] | [1, 2250, 7168] | Input to attention (MLA) |
| `layer_N_attn_output` | [1, 10, 7168] | [1, 2250, 7168] | Output from attention |
| `layer_N_indexer_input` | [1, 10, 7168] | [1, 2250, 7168] | Input to Lightning Indexer |
| `layer_N_indexer_wk_output` | [1, 10, 128] | [1, 2250, 128] | Indexer's K projection |
| `layer_N_indexer_topk_indices` | [1, 10, **10**] | [1, 2250, **2048**] | **Top-k sparse indices** |
| `layer_N_wo_output` | [1, 10, 7168] | [1, 2250, 7168] | Attention output projection |
| `layer_N_ffn_norm_output` | [1, 10, 7168] | [1, 2250, 7168] | Pre-FFN layer norm |
| `layer_N_ffn_input` | [1, 10, 7168] | [1, 2250, 7168] | Input to FFN/MoE |
| `layer_N_ffn_output` | [1, 10, 7168] | [1, 2250, 7168] | Output from FFN/MoE |
| `final_norm_output` | [1, 10, 7168] | [1, 2250, 7168] | Final layer norm output |
| `logits` | [1, 16160] | [1, 16160] | LM head output (sharded) |

**Key difference:** For `topk_indices`:
- **prompt_0-4:** Shape is `[1, seq_len, seq_len]` - all tokens selected (dense)
- **prompt_5:** Shape is `[1, 2250, 2048]` - only 2048 of 2250 tokens selected (**sparse!**)

### File Naming Convention

```
{step:04d}_rank{rank}_{tensor_name}.pt
```

Example: `0000_rank3_layer_2_indexer_topk_indices.pt`

### Disk Organization

```
/mnt/models-disk/official_tensors/           (2.2GB total)
│
├── prompt_0_simple_math/                    (424 files, ~95MB)
│   ├── 0000_rank0_embedding_output.pt
│   ├── 0000_rank0_layer_0_attn_input.pt
│   ├── 0000_rank0_layer_0_attn_output.pt
│   ├── 0000_rank0_layer_0_indexer_topk_indices.pt
│   ├── 0000_rank0_layer_0_*.pt              (10 tensors per layer)
│   ├── 0000_rank0_layer_1_*.pt              (layers 1-4)
│   ├── 0000_rank0_final_norm_output.pt
│   ├── 0000_rank0_logits.pt
│   ├── 0000_rank1_*.pt                      (same 53 tensors for rank 1)
│   └── 0000_rank7_*.pt                      (same 53 tensors for rank 7)
│
├── prompt_1_greeting/                       (424 files)
├── prompt_2_code_generation/                (424 files)
├── prompt_3_explanation/                    (424 files)
└── prompt_4_long_context/                   (424 files, larger tensors)
```

**File naming:** `0000_rank{rank}_{tensor_name}.pt`
- `rank` = tensor parallel shard (0-7)
- `tensor_name` = e.g., `layer_0_attn_output`

**Per-directory breakdown (424 files each):**
- 53 unique tensors × 8 ranks = 424 files
- 53 tensors = 1 embedding + 50 layer tensors + 1 final_norm + 1 logits

**To re-run captures:** `bash ~/deepseek-v3.2-inference/capture_all_prompts.sh`

### Loading Captured Tensors

```python
import torch

# Load a single tensor from a specific prompt
prompt_dir = "/mnt/models-disk/official_tensors/prompt_0_simple_math"
tensor_data = torch.load(f"{prompt_dir}/0000_rank0_layer_0_attn_output.pt")
print(f"Name: {tensor_data['name']}")
print(f"Shape: {tensor_data['shape']}")
print(f"Dtype: {tensor_data['dtype']}")
print(f"Mean: {tensor_data['mean']:.6f}")
print(f"Data: {tensor_data['data']}")  # The actual tensor (float32)

# Load all ranks for a tensor and concatenate
def load_all_ranks(tensor_name, prompt_dir):
    """Load tensor from all 8 ranks."""
    tensors = []
    for rank in range(8):
        path = f"{prompt_dir}/0000_rank{rank}_{tensor_name}.pt"
        data = torch.load(path)
        tensors.append(data['data'])
    # Note: How to concatenate depends on the tensor type
    # For column-parallel: torch.cat(tensors, dim=-1)
    # For row-parallel: tensors are identical, use any one
    return tensors

# Example: Load indexer indices for long context prompt
long_ctx_dir = "/mnt/models-disk/official_tensors/prompt_4_long_context"
indices = load_all_ranks("layer_0_indexer_topk_indices", long_ctx_dir)
print(f"Indexer indices shape: {indices[0].shape}")  # [1, 188, topk]
```

### Re-running Tensor Capture

The capture script is at `~/deepseek-v3.2-inference/capture_tensors.py` on rn1h:

**Short prompt (inline):**
```bash
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --tunnel-through-iap --command='
cd ~/deepseek-v3.2-inference
unset NCCL_SOCKET_IFNAME
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
    --config config_671B_v3.2.json \
    --output-dir /mnt/models-disk/official_tensors/prompt_0_simple_math \
    --layer-limit 5 \
    --prompt "What is 2+2?"
'
```

**Long prompt (from file) - triggers sparse attention:**
```bash
# First copy the prompt file to cluster
gcloud compute scp debug_v32/long_prompt_sparse.txt h200-mig-cluster-rn1h:~/deepseek-v3.2-inference/ \
    --zone=europe-west1-b --tunnel-through-iap

# Then run capture with --prompt-file
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --tunnel-through-iap --command='
cd ~/deepseek-v3.2-inference
unset NCCL_SOCKET_IFNAME
torchrun --nproc-per-node=8 capture_tensors.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
    --config config_671B_v3.2.json \
    --output-dir /mnt/models-disk/official_tensors/prompt_5_sparse_trigger \
    --layer-limit 5 \
    --prompt-file long_prompt_sparse.txt
'
```

The script will print whether sparse attention is triggered:
```
Token count: 2250
  -> SPARSE ATTENTION TRIGGERED: seq_len (2250) > index_topk (2048)
     topk_indices will have shape [1, 2250, 2048]
```

**All prompts at once:**
```bash
gcloud compute scp debug_v32/capture_all_prompts.sh debug_v32/long_prompt_sparse.txt \
    h200-mig-cluster-rn1h:~/deepseek-v3.2-inference/ --zone=europe-west1-b --tunnel-through-iap

gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --tunnel-through-iap --command='
cd ~/deepseek-v3.2-inference && bash capture_all_prompts.sh
'
```

### Debugging with Captured Tensors

1. **Compare embedding output**: First check if embeddings match between official and HF fork
2. **Check indexer inputs/outputs**: `indexer_input` → `indexer_topk_indices` flow
3. **Verify attention output**: Compare `attn_output` layer by layer
4. **Check FFN/MoE**: Compare `ffn_input` → `ffn_output`

If divergence is found, the first differing tensor indicates the root cause location.

---

### How to Use Reference Outputs for Debugging

1. **Run HF fork with same prompts:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("/models-local/DeepSeek-V3.2-fp8", ...)
   tokenizer = AutoTokenizer.from_pretrained("/models-local/DeepSeek-V3.2-fp8")

   prompts = ["What is 2+2?", "Hello, how are you?", ...]
   for prompt in prompts:
       inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], ...)
       outputs = model.generate(inputs, max_new_tokens=200, temperature=0.0)
       print(tokenizer.decode(outputs[0]))
   ```

2. **Compare outputs:**
   - If HF fork produces gibberish but official produces coherent text → Bug in HF fork
   - If both produce similar output → Problem may be elsewhere
   - If HF fork is close but slightly different → Check numerical precision

3. **Identify divergence point:**
   - Use tensor comparison (see Step 4 in debug workflow)
   - Check Indexer outputs specifically for Prompt 5 (long context)

---

### BF16 Path (Optional)

```
Step 1: Dequantize FP8 → BF16 (HuggingFace format)
┌─────────────────────────────────────────────────────────────────────┐
│ Input:  /models/DeepSeek-V3.2-fp8 (HF FP8, 643GB)                   │
│ Script: convert_fp8_to_bf16.py                                      │
│ Output: /models-local/DeepSeek-V3.2-bf16 (HF BF16, ~1.3TB)          │
│ Time:   ~45 min (163 shards × ~16s each)                            │
│                                                                     │
│ What it does:                                                       │
│   - Reads FP8 weights (torch.float8_e4m3fn)                         │
│   - Reads scale tensors (weight_scale_inv)                          │
│   - Computes: bf16_weight = fp8_weight × scale_inv                  │
│   - Saves as standard BF16 safetensors                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Reshard for tensor parallelism (Official format)
┌─────────────────────────────────────────────────────────────────────┐
│ Input:  /models-local/DeepSeek-V3.2-bf16 (HF BF16)                  │
│ Script: convert.py --model-parallel 8                               │
│ Output: /models-local/DeepSeek-V3.2-bf16-mp8 (8 shards)             │
│ Time:   ~15 min                                                     │
│                                                                     │
│ What it does:                                                       │
│   - Renames HF param names → official names                         │
│   - Shards weights across MP ranks (dim 0 or 1)                     │
│   - Distributes experts across ranks                                │
│   - PRESERVES dtype (BF16 in → BF16 out)                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: Run official inference
┌─────────────────────────────────────────────────────────────────────┐
│ Checkpoint: /models-local/DeepSeek-V3.2-bf16-mp8                    │
│ Config:     config_671B_v3.2_bf16.json (dtype: "bf16")              │
│ Command:                                                            │
│   torchrun --nproc-per-node=8 generate.py \                         │
│       --ckpt-path /models-local/DeepSeek-V3.2-bf16-mp8 \            │
│       --config config_671B_v3.2_bf16.json \                         │
│       --input-file test_prompts.txt \                               │
│       --max-new-tokens 100                                          │
│                                                                     │
│ Why BF16 works:                                                     │
│   - Linear layers use standard F.linear() (no tilelang)             │
│   - Indexer still uses fp8_index but may have fallback              │
│   - If Indexer fails, may need to patch official code               │
└─────────────────────────────────────────────────────────────────────┘
```

### Config Files

**FP8 config (config_671B_v3.2.json):**
```json
{
    "dtype": "fp8",
    "scale_fmt": "ue8m0",
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    ...
}
```

**BF16 config (config_671B_v3.2_bf16.json):**
```json
{
    "dtype": "bf16",
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    ...
}
```

### Key Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `convert_fp8_to_bf16.py` | `~/debug_v32/` | Dequantize FP8 → BF16 |
| `convert.py` | Official inference repo | Reshard HF → Official format |
| `generate.py` | Official inference repo | Run inference |

---

## Problem Statement

The DeepSeek V3.2 HuggingFace fork at `shuyingl/deepseek-v32-minimal-on-v4.57.3` generates **gibberish output**, while DeepSeek V3, V2, and V2-lite work correctly. This isolates the problem to the V3.2-specific implementation.

### What's Different in V3.2

V3.2 adds **one architectural change** over V3: the **Lightning Indexer** for sparse attention.

| Component | V3 | V3.2 |
|-----------|----|----|
| Attention | Dense (all tokens) | Sparse (top-k via Indexer) |
| Indexer | None | `DeepseekV32Indexer` with Hadamard transform |
| RoPE in Indexer | N/A | Non-interleaved (like Llama) |
| Config | `deepseek_v3` | `deepseek_v32` with 7 new params |

---

## Cluster Information

**Current: 1 node (8 H200 GPUs) after resize**

| Node | Name | Internal IP | Status |
|------|------|-------------|--------|
| 0 | h200-mig-cluster-rn1h | 10.0.0.17 | ✅ Active |

**Previous nodes (deleted during resize):**
- ~~h200-mig-cluster-m019~~ (was master, had NFS)
- ~~h200-mig-cluster-8slj~~
- ~~h200-mig-cluster-lgsn~~
- ~~h200-mig-cluster-fm3h~~

**Key Paths on Cluster:**
- FP8 Checkpoint: `/models-local/DeepSeek-V3.2-fp8` (persistent disk, fast)
- V3 BF16: `/models-local/DeepSeek-V3-bf16` (persistent disk)
- MP=8 Official: `/models-local/DeepSeek-V3.2-converted-mp8` ✅
- Debug scripts: Need to re-setup on rn1h

**Storage Notes:**
- `/models-local/` → `/mnt/models-disk` (persistent disk, survives resize!)
- `/models/` → `/mnt/gcs/models` (GCS bucket via gcsfuse, slow)

**SSH Access:**
```bash
gcloud compute ssh h200-mig-cluster-rn1h --zone=europe-west1-b --project=fundamental-labs
```

---

## IMPORTANT: File Transfer Rules

**DO NOT** try to send large files (checkpoints, model weights) from local to cluster. Use:
- GCS buckets (`gs://fundamental_ml_shared_storage/`)
- Cluster local disk (`/models/`, `/data/`)

**DO** use short SCP commands for small scripts (a few KB each).

---

## Debug Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEBUGGING WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Copy scripts to cluster (from local machine)                       │
│  ─────────────────────────────────────────────────────                      │
│  gcloud compute scp --recurse debug_v32 h200-mig-cluster-sl5q:~/            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 2: Verify OFFICIAL code + checkpoint work (GROUND TRUTH)              │
│  ─────────────────────────────────────────────────────────────              │
│  • Download official inference code from HuggingFace                        │
│  • Convert HF checkpoint to official format (required!)                     │
│  • Run official inference → verify sensible output                          │
│                                                                              │
│  If this fails → Problem is checkpoint, not HF fork!                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 3: Test HF Fork (Quick Sanity Check)                                  │
│  ─────────────────────────────────────────                                  │
│  Tests: config, weights, dense path, sparse path, generation                │
│                                                                              │
│  Results:                                                                    │
│    • DENSE fails → Problem in base V3 code or weight loading                │
│    • SPARSE fails but DENSE works → Problem in Indexer                      │
│    • Both fail → Likely config or weight mapping issue                      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 4: Tensor Comparison (if needed)                                      │
│  ─────────────────────────────────────                                      │
│  Run official inference → save tensors (ground truth)                       │
│  Run HF fork → save same tensors                                            │
│  Compare tensor-by-tensor to find EXACT divergence point                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Copy Debug Scripts to Cluster

The debug scripts are in this local directory. Copy them to the cluster master node:

```bash
# From your LOCAL machine, copy the debug_v32 folder to cluster
# This is a small folder (~50KB of Python scripts)

gcloud compute scp --recurse \
    /path/to/transformers/.conductor/seville-v3/debug_v32 \
    h200-mig-cluster-m019:~/debug_v32 \
    --zone=europe-west1-b --project=fundamental-labs

# Verify files arrived (short command, not streaming)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs \
    -- "ls -la ~/debug_v32/"
```

**Files being copied (~50KB total):**
- `quick_sanity_check.py` - Fast single-GPU diagnostics
- `1_setup_official.sh` - Download official inference code
- `2_instrument_official.py` - Patch official code
- `3_run_official.sh` - Run official (ground truth)
- `4_run_hf_fork.py` - Run HF fork with tensor saving
- `5_compare_tensors.py` - Compare tensors
- `README.md` - This file

---

## Step 2: Verify Official Inference Works (CRITICAL!)

**Why this step matters:** Before debugging the HF fork, we must verify the official DeepSeek inference code produces correct output with the checkpoint. If official code fails too, the problem is the checkpoint, not your fork.

### 2a. Download Official Inference Code

```bash
# SSH to master
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# === ON THE CLUSTER ===
source /etc/profile.d/nccl.sh
cd ~/debug_v32
mkdir -p official_inference
cd official_inference

# Download official inference files from HuggingFace
for file in model.py generate.py convert.py kernel.py config_671B_v3.2.json requirements.txt; do
    curl -sL "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/inference/$file" -o "$file"
    echo "Downloaded $file"
done

# Install dependencies
pip install -r requirements.txt
pip install fast-hadamard-transform
```

### 2b. Convert Checkpoint (REQUIRED!)

The official inference code uses a **different checkpoint format** than HuggingFace. You must convert first:

**MP (Model Parallel) Constraints:**
| MP Value | Status | Notes |
|----------|--------|-------|
| 8 | ✅ Works | Single node (8 GPUs) |
| 16 | ✅ Works | 2 nodes (16 GPUs) - **RECOMMENDED** |
| 32 | ❌ Fails | FP8 scale dimensions not divisible by 32 |
| 40 | ❌ Fails | 256 experts not divisible by 40 |

MP must satisfy:
1. `256 % MP == 0` (experts divisible by MP)
2. All sharded weight dimensions divisible by MP
3. **FP8 checkpoint quirk:** The convert.py shards FP8 scales (`weight_scale_inv`) using parent weight's dim.
   FP8 scales have dims like 144, 16, 56 → divisible by 16 but not 32.
   A BF16 checkpoint might support MP=32.

```bash
# === ON THE CLUSTER ===
cd ~/debug_v32/official_inference

# Convert HF checkpoint to official format
# This shards the weights for tensor parallelism
export HF_CKPT="/models-local/DeepSeek-V3.2-fp8"  # Use fast NFS disk
export CONVERTED_CKPT="/models-local/DeepSeek-V3.2-converted"
export MP=16  # Use 16 GPUs (2 nodes) - max tested working value

python convert.py \
    --hf-ckpt-path "$HF_CKPT" \
    --save-path "$CONVERTED_CKPT" \
    --n-experts 256 \
    --model-parallel $MP

# This creates files like:
#   /models-local/DeepSeek-V3.2-converted/model0-mp16.safetensors
#   /models-local/DeepSeek-V3.2-converted/model1-mp16.safetensors
#   ... (16 shards total)
```

**Note:** Conversion takes time and disk space. The converted checkpoint will be saved to `/models-local/DeepSeek-V3.2-converted` (NFS, shared across nodes). Once verified working, you can copy to GCS for persistence.

**Parallel Conversion Strategy:** Run both MP=8 and MP=16 conversions simultaneously (CPU-only, no GPU conflict):
```bash
# MP=16 (2 nodes) - for multi-node inference
nohup python3 convert.py --hf-ckpt-path /models-local/DeepSeek-V3.2-fp8 \
    --save-path /models-local/DeepSeek-V3.2-converted --n-experts 256 --model-parallel 16 \
    > /tmp/convert_mp16.log 2>&1 &

# MP=8 (single node) - for parallel single-node jobs
nohup python3 convert.py --hf-ckpt-path /models-local/DeepSeek-V3.2-fp8 \
    --save-path /models-local/DeepSeek-V3.2-converted-mp8 --n-experts 256 --model-parallel 8 \
    > /tmp/convert_mp8.log 2>&1 &
```

### 2c. Run Official Inference

**Option 1: Single-node with MP=8 (SIMPLER)**
```bash
# === ON THE CLUSTER (master node) ===
cd ~/debug_v32/official_inference

# Run with 8 GPUs on single node
torchrun --nproc-per-node=8 generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
    --config config_671B_v3.2.json \
    --max-new-tokens 50 \
    --temperature 0.0 \
    --interactive
```

**Option 2: Multi-node with MP=16 (requires 2 nodes)**
```bash
# === Run on BOTH nodes simultaneously ===
# Node 0 (master) - 10.0.0.13
cd ~/debug_v32/official_inference
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=10.0.0.13 --master-port=29500 \
    generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted \
    --config config_671B_v3.2.json \
    --max-new-tokens 50 --temperature 0.0

# Node 1 (worker) - 10.0.0.14 (run in separate terminal)
cd ~/debug_v32/official_inference
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
    --master-addr=10.0.0.13 --master-port=29500 \
    generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted \
    --config config_671B_v3.2.json \
    --max-new-tokens 50 --temperature 0.0
```

**Note:** `nproc-per-node × nnodes` must match `--model-parallel` used during conversion.

Then type a test prompt like: `Hello, how are you today?`

### 2d. Expected Result

**If official code works:** You should see coherent, sensible responses.
```
User: Hello, how are you today?
Assistant: Hello! I'm doing well, thank you for asking. How can I help you today?
```

**If official code fails/produces gibberish:** The problem is likely:
- Checkpoint is corrupted or incomplete
- Wrong checkpoint (V3 instead of V3.2)
- Conversion failed
- Missing dependencies

**DO NOT proceed to Step 3 until official code works!**

### Fire-and-Forget Version (Single-node MP=8)

```bash
# From LOCAL - run in background on single node
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "cd ~/debug_v32/official_inference && source /etc/profile.d/nccl.sh && \
     echo 'Hello, how are you?' | torchrun --nproc-per-node=8 generate.py \
         --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
         --config config_671B_v3.2.json \
         --max-new-tokens 50 --temperature 0.0 \
     > /tmp/official_test.log 2>&1"

# Check result later
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b -- \
    "cat /tmp/official_test.log"
```

---

## Current Node Reservations

**⚠️ CHECK BEFORE USING NODES:**
| Node | Name | Status | Job | Port |
|------|------|--------|-----|------|
| 0 | m019 | 🔴 RESERVED | MP=16 Official Inference (master) | 29500 |
| 1 | 8slj | 🔴 RESERVED | MP=16 Official Inference (worker) | 29500 |
| 2 | lgsn | 🟢 AVAILABLE | - | - |
| 3 | fm3h | 🟢 AVAILABLE | - | - |
| 4 | rn1h | 🟢 AVAILABLE | - | - |

**Check if jobs are still running:**
```bash
# From master - check all nodes
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "ps aux | grep -E 'generate.py|torchrun' | grep -v grep && \
     ssh 10.0.0.14 'ps aux | grep -E \"generate.py|torchrun\" | grep -v grep' 2>/dev/null"
```

**Update this table when reserving/releasing nodes!**

---

## Parallel Debugging with 40 GPUs

With 5 nodes × 8 GPUs = 40 GPUs, you can run multiple tasks simultaneously:

### Converted Checkpoints Available
| Path | MP | GPUs | Use Case |
|------|-----|------|----------|
| `/models-local/DeepSeek-V3.2-converted` | 16 | 16 (2 nodes) | Multi-node official inference |
| `/models-local/DeepSeek-V3.2-converted-mp8` | 8 | 8 (1 node) | Single-node parallel jobs |

### Parallel Debugging Strategy

**Option A: Split by task type (RECOMMENDED)**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    40 GPU PARALLEL DEBUGGING                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Node 0 (8 GPUs): Official inference MP=8 → save tensors               │
│  Node 1 (8 GPUs): Official inference MP=8 → different prompts          │
│  Node 2 (8 GPUs): HF fork testing → save tensors                       │
│  Node 3 (8 GPUs): HF fork testing → different configs                  │
│  Node 4 (8 GPUs): Tensor comparison / spare                            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Option B: Run official (16 GPU) + HF fork (24 GPU) in parallel**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  Nodes 0-1 (16 GPUs): Official inference MP=16                         │
│  Nodes 2-4 (24 GPUs): HF fork with device_map="auto"                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Running Parallel Jobs on Specific Nodes

**Official inference on Node 0 only (MP=8):**
```bash
# SSH to specific node
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# Run on Node 0
cd ~/debug_v32/official_inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
    --config config_671B_v3.2.json \
    --max-new-tokens 50 --temperature 0.0
```

**HF fork on Node 2 (via SSH to worker):**
```bash
# SSH to worker node
gcloud compute ssh h200-mig-cluster-lgsn --zone=europe-west1-b --project=fundamental-labs

# Run HF fork
cd ~/debug_v32
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python quick_sanity_check.py \
    --checkpoint /models-local/DeepSeek-V3.2-fp8
```

### Parallel Tensor Comparison Workflow

Run both official and HF fork simultaneously, then compare:

```bash
# Terminal 1 - Node 0: Official inference with tensor saving
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "cd ~/debug_v32/official_inference && \
     DEBUG_SAVE_TENSORS=1 DEBUG_OUTPUT_DIR=/tmp/tensors_official \
     torchrun --nproc-per-node=8 generate.py \
         --ckpt-path /models-local/DeepSeek-V3.2-converted-mp8 \
         --config config_671B_v3.2.json --max-new-tokens 20 \
     > /tmp/official_run.log 2>&1 &"

# Terminal 2 - Node 2: HF fork with tensor saving (in parallel)
gcloud compute ssh h200-mig-cluster-lgsn --zone=europe-west1-b --project=fundamental-labs -- \
    "cd ~/debug_v32 && \
     python 4_run_hf_fork.py \
         --checkpoint /models-local/DeepSeek-V3.2-fp8 \
         --output-dir /tmp/tensors_hf_fork \
     > /tmp/hf_fork_run.log 2>&1 &"

# Terminal 3 - Compare when both complete
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "python ~/debug_v32/5_compare_tensors.py \
         --official /tmp/tensors_official \
         --fork /tmp/tensors_hf_fork"
```

---

## Step 3: Test HF Fork (Quick Sanity Check)

**Only do this after Step 2 succeeds!**

### Prerequisites

The HF fork code needs to be available on the cluster. Options:
1. `pip install git+https://github.com/huggingface/transformers@shuyingl/deepseek-v32-minimal-on-v4.57.3`
2. Or clone and install: `git clone ... && pip install -e .`

### Run Sanity Check

```bash
# SSH to master
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# === ON THE CLUSTER ===
source /etc/profile.d/nccl.sh
cd ~/debug_v32

# Install your fork if not already installed
pip install git+https://github.com/huggingface/transformers@shuyingl/deepseek-v32-minimal-on-v4.57.3

# Run sanity check (uses single GPU for config/weight checks, multi-GPU for inference)
python quick_sanity_check.py --checkpoint /models-local/DeepSeek-V3.2-fp8
```

### Fire-and-Forget Version

```bash
# From LOCAL, send command to run in background
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "cd ~/debug_v32 && source /etc/profile.d/nccl.sh && \
     nohup python quick_sanity_check.py --checkpoint /models-local/DeepSeek-V3.2-fp8 \
     > /tmp/sanity_check.log 2>&1 &"

# Check results later (snapshot, not streaming!)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "cat /tmp/sanity_check.log"
```

### What It Checks

1. **Config Verification**
   - `model_type == "deepseek_v32"` (not `deepseek_v3`)
   - V3.2 params: `index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`

2. **Weight Loading**
   - Indexer weights exist: `wq_b`, `wk`, `k_norm`, `weights_proj` per layer
   - Weights are non-zero (std > 1e-8)

3. **Forward Pass (Dense)**
   - Set `use_sparse_attention=False` → uses V3 code path
   - Check for NaN/Inf in logits
   - Verify prediction on "The capital of France is" → should predict "Paris"

4. **Forward Pass (Sparse)**
   - Set `use_sparse_attention=True` → uses Indexer
   - Same checks as dense

5. **Generation Test**
   - Check for gibberish, excessive repetition

### Interpreting Results

| Symptom | Likely Cause |
|---------|--------------|
| No indexer weights | Checkpoint is V3 not V3.2, or config.model_type wrong |
| Zero indexer weights | Weight initialization issue |
| Dense fails | Base V3 inheritance broken or weight loading issue |
| Sparse fails, Dense works | Indexer implementation bug |
| Both produce gibberish | Fundamental weight/config mismatch |

---

## Step 4: Full Tensor Comparison (If Needed)

**Only do this if:**
- Step 2 passed (official works)
- Step 3 failed (HF fork doesn't work)

This compares intermediate activations between official and HF fork to find exactly where outputs diverge.

### 4a. Instrument Official Code

```bash
# SSH to master
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# === ON THE CLUSTER ===
cd ~/debug_v32
python 2_instrument_official.py
```

This patches `official_inference/model.py` to save tensors at key checkpoints.

### 4b. Run Both with Tensor Saving

```bash
# === ON THE CLUSTER ===

# Run official (ground truth)
export DEBUG_SAVE_TENSORS=1
export DEBUG_OUTPUT_DIR="$HOME/debug_v32/saved_tensors/official"
export DEBUG_LAYER_LIMIT=3

cd ~/debug_v32/official_inference
torchrun --nproc-per-node=16 generate.py \
    --ckpt-path /models-local/DeepSeek-V3.2-converted \
    --config config_671B_v3.2.json \
    --max-new-tokens 20 --temperature 0.0

# Run HF fork
cd ~/debug_v32
python 4_run_hf_fork.py \
    --checkpoint /models-local/DeepSeek-V3.2-fp8 \
    --output-dir ~/debug_v32/saved_tensors/hf_fork
```

### 4c. Compare Tensors

```bash
python 5_compare_tensors.py \
    --official ~/debug_v32/saved_tensors/official \
    --fork ~/debug_v32/saved_tensors/hf_fork
```

### Tensor Checkpoints Compared

| Checkpoint | What It Tests |
|------------|---------------|
| `embedding_output` | Embedding layer weights |
| `indexer_L0_input_x` | Hidden states entering indexer |
| `indexer_L0_input_qr` | Compressed query (q_a_layernorm output) |
| `indexer_L0_q_after_rope` | RoPE application (non-interleaved) |
| `indexer_L0_k_after_rope` | Key RoPE + k_norm |
| `indexer_L0_q_after_hadamard` | Hadamard transform |
| `indexer_L0_index_score` | Score computation |
| `indexer_L0_topk_indices` | Top-k selection |
| `mla_L0_output` | Attention output with sparse mask |
| `layer_L0_output` | Full layer output (incl. MoE) |
| `final_logits` | LM head output |

---

## Key Implementation Details

### Indexer Score Computation

**Official (model.py):**
```python
weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
index_score = fp8_index(q_fp8, weights, k_fp8, k_scale)  # Custom kernel
```

**HF Fork (modular_deepseek_v32.py:~line 499):**
```python
weights = self.weights_proj(hidden_states) * (self.num_heads**-0.5) * self.softmax_scale
weights = weights.transpose(1, 2).unsqueeze(-1)  # [B, H, S, 1]
scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, S_q, S_k]
scores = F.relu(scores)
index_scores = (scores * weights).sum(dim=1)  # [B, S_q, S_k]
```

### RoPE Differences

- **MLA (main attention)**: Uses INTERLEAVED RoPE
- **Indexer**: Uses NON-INTERLEAVED RoPE (like Llama)

### Sparse Mask Creation

```python
# Create mask with -inf everywhere
sparse_mask = torch.full((batch, seq_len, kv_seq_len), float("-inf"))

# Set selected positions to 0
sparse_mask.scatter_(-1, topk_indices, 0.0)

# Combine with causal mask
sparse_mask = sparse_mask + attention_mask
```

---

## Likely Root Causes (Priority Order)

### 1. Indexer Weights Not Loaded
**Check:** Run `quick_sanity_check.py` - it reports if indexer params are missing or zero.

**Fix:** Ensure checkpoint's `config.json` has `model_type: "deepseek_v32"`

### 2. Hadamard Transform Missing/Wrong
**Check:** Look for import errors or fallback to slow path.

**Fix:** Install `pip install fast-hadamard-transform`

### 3. RoPE Interleave Mismatch
**Check:** Compare `indexer_L0_q_after_rope` tensors.

**Fix:** Verify indexer uses `apply_rotary_pos_emb` (non-interleaved)

### 4. Index Score Scaling
**Check:** Compare `indexer_L0_index_score` tensors.

**Fix:** Verify scaling factors: `n_heads ** -0.5 * softmax_scale`

### 5. Config Inheritance
**Check:** Verify all V3 config params are inherited correctly.

**Fix:** Check `DeepseekV32Config.__init__` passes all params to `super().__init__`

---

## Debug Scripts Reference

| Script | Purpose |
|--------|---------|
| `quick_sanity_check.py` | Fast single-GPU check of config, weights, forward pass |
| `1_setup_official.sh` | Download official inference code (or do manually per Step 2) |
| `2_instrument_official.py` | Patch official code to save tensors |
| `3_run_official.sh` | Run official code (ground truth) |
| `4_run_hf_fork.py` | Run HF fork with tensor saving |
| `5_compare_tensors.py` | Compare saved tensors, find divergence |

---

## Fire-and-Forget Commands Reference

**IMPORTANT:** Never use `tail -f` or streaming commands. Always use snapshots.

```bash
# Check cluster status (from local)
gcloud compute instances list --filter="name~h200-mig-cluster" --project=fundamental-labs

# SSH to master (short session)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs

# Copy scripts to cluster (from local, ~50KB)
gcloud compute scp --recurse debug_v32 h200-mig-cluster-m019:~/ \
    --zone=europe-west1-b --project=fundamental-labs

# Run command in background (from local)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "cd ~/debug_v32 && nohup python quick_sanity_check.py --checkpoint /models-local/DeepSeek-V3.2-fp8 > /tmp/output.log 2>&1 &"

# Check logs later - SNAPSHOT not streaming! (from local)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "tail -100 /tmp/output.log"

# Kill running processes (from local)
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "pkill -9 -f python"
```

---

## Quick Test Toggle

To quickly test if the issue is in the Indexer vs base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/models-local/DeepSeek-V3.2-fp8", ...)

# Test 1: Dense (V3 path) - should work if V3 works
model.config.use_sparse_attention = False
outputs = model.generate(...)

# Test 2: Sparse (Indexer path) - this is likely broken
model.config.use_sparse_attention = True
outputs = model.generate(...)
```

---

## Development Workflow

### Branch Information

**Always use this branch for testing:**
```
Branch: shuyingl/deepseek-v32-minimal-on-v4.57.3
Remote: origin (lyfegame/transformers)
```

### Updating Code on Cluster

When you make local changes and want them available on the cluster:

```bash
# 1. Make sure you're on the correct branch
git checkout shuyingl/deepseek-v32-minimal-on-v4.57.3

# 2. If you edited modular_deepseek_v32.py, regenerate the modeling file
python utils/modular_model_converter.py --files_to_parse src/transformers/models/deepseek_v32/modular_deepseek_v32.py

# 3. Commit and push
git add src/transformers/models/deepseek_v32/
git commit -m "Your commit message"
git push origin shuyingl/deepseek-v32-minimal-on-v4.57.3

# 4. Update on cluster
gcloud compute ssh h200-mig-cluster-m019 --zone=europe-west1-b --project=fundamental-labs -- \
    "pip install --user --force-reinstall git+https://github.com/lyfegame/transformers@shuyingl/deepseek-v32-minimal-on-v4.57.3"
```

### Debug Logging

The Indexer has built-in debug logging controlled by environment variable:

```bash
# Enable debug logging for layer 0 indexer
export DEBUG_DEEPSEEK_INDEXER=1
python your_script.py
```

This logs shapes, statistics, and intermediate values at each step of the indexer computation.

---

## Reference Links

- HF Fork: `src/transformers/models/deepseek_v32/modular_deepseek_v32.py`
- Official inference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/inference
- Checkpoint on cluster: `/models-local/DeepSeek-V3.2-fp8` (fast) or `/models/DeepSeek-V3.2-fp8` (GCS)
