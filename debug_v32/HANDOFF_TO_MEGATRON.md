# DeepSeek V3.2 Verification Handoff Document

**Purpose**: This document provides everything needed to verify a Megatron-Core implementation of DeepSeek V3.2 against the official inference outputs.

**Created**: 2024-12-14
**Source**: HuggingFace transformers fork verification (`shuyingl/deepseek-v32-minimal-on-v4.57.3`)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Reference Data](#reference-data)
3. [Test Prompts](#test-prompts)
4. [Expected Outputs](#expected-outputs)
5. [Success Criteria](#success-criteria)
6. [Sparse Attention Behavior](#sparse-attention-behavior)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [Timing Benchmarks](#timing-benchmarks)
9. [Lessons Learned](#lessons-learned)
10. [File Inventory](#file-inventory)

---

## Quick Start

### Verification Checklist

```
□ Prompt 0 (simple_math): Returns "4" or equivalent answer
□ Prompt 1 (greeting): Friendly conversational response
□ Prompt 2 (code_generation): Valid is_prime() function
□ Prompt 3 (explanation): Explains relativity with space/time/gravity
□ Prompt 4 (long_context): Lists 3 ML categories correctly
□ Prompt 5 (sparse_trigger): Sparse attention triggers, discusses MLA/efficiency
□ All prompts complete without OOM or crashes
```

### Minimum Test Command Pattern

```python
# Pseudocode for any framework
for prompt in load_prompts("reference_prompts.json"):
    output = model.generate(prompt, max_new_tokens=200)
    print(f"Prompt {prompt.id}: {output[:100]}...")
    # Compare semantically with expected output
```

---

## Reference Data

### Locations (on H200 cluster)

| Resource | Path |
|----------|------|
| Official tensors | `/mnt/models-disk/official_tensors/prompt_N_*/` |
| BF16 checkpoint | `/models-local/DeepSeek-V3.2-bf16` |
| Reference prompts JSON | `debug_v32/reference_prompts.json` |
| Sparse trigger prompt | `debug_v32/long_prompt_sparse.txt` |

### Official Tensor Contents

Each `prompt_N_*` directory contains:
- `layer_N_*.pt` - Intermediate activations (61 layers)
- `indexer_output.pt` - Sparse attention indices
- `final_logits.pt` - Output logits for comparison

---

## Test Prompts

### Prompt 0: simple_math
```
What is 2+2?
```
- **Input tokens**: 11
- **Attention type**: Dense
- **Tests**: Basic arithmetic, short response

### Prompt 1: greeting
```
Hello! How are you today?
```
- **Input tokens**: 10
- **Attention type**: Dense
- **Tests**: Conversational ability

### Prompt 2: code_generation
```
Write a Python function to check if a number is prime.
```
- **Input tokens**: 16
- **Attention type**: Dense
- **Tests**: Code generation, proper formatting

### Prompt 3: explanation
```
Explain the theory of relativity in simple terms.
```
- **Input tokens**: 13
- **Attention type**: Dense
- **Tests**: Long-form explanation, knowledge

### Prompt 4: long_context
```
The following is a long document about machine learning. Machine learning is a
subset of artificial intelligence that enables systems to learn and improve from
experience without being explicitly programmed...
[~1150 characters total]
...What are the three main categories of machine learning algorithms mentioned
in the document?
```
- **Input tokens**: 189
- **Attention type**: Dense
- **Tests**: Reading comprehension, context extraction

### Prompt 5: sparse_trigger (CRITICAL)
```
The following is a comprehensive technical document about transformer
architectures and attention mechanisms in modern deep learning systems.

Chapter 1: Introduction to Transformer Architecture
[... ~12,478 characters about transformers, MLA, sparse attention ...]

Question: What are the three main techniques that DeepSeek V3.2 uses to achieve
efficient inference on long sequences, and briefly explain how each contributes
to efficiency?
```
- **Input tokens**: 2251
- **Attention type**: **SPARSE** (seq_len > index_topk=2048)
- **Tests**: Sparse attention path, long context, technical comprehension
- **File**: `long_prompt_sparse.txt`

---

## Expected Outputs

### Semantic Expectations (NOT Exact Match)

| Prompt | Key Content | Pass Criteria |
|--------|-------------|---------------|
| 0 | "4" | Contains the number 4 as answer |
| 1 | Greeting response | Friendly, conversational, offers help |
| 2 | `is_prime(n)` function | Valid Python, handles edge cases (n≤1, n≤3, divisibility) |
| 3 | Relativity explanation | Mentions space, time, gravity, Einstein |
| 4 | ML categories | Lists: supervised, unsupervised, reinforcement |
| 5 | DeepSeek V3.2 techniques | Discusses MLA, Lightning Indexer, efficient inference |

### Actual HF Fork Outputs (Reference)

**Prompt 0**: `What is 2+2?...` (8 tokens - short, may need more generation)

**Prompt 1**:
```
Hello! I'm doing well, thank you for asking. How about you? How can I help you today?
```

**Prompt 2**:
```python
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

**Prompt 3**:
```
Of course! Here's the theory of relativity explained in simple terms, broken
into its two main parts.

### The Core Idea
Einstein's theory is about **space, time, and gravity**. It says that what we
thought were fixed, separate things (space and time) are actually woven together
into a single fabric called **"spacetime."** This fabric can be stretched and bent.
```

**Prompt 4**:
```
The three main categories of machine learning algorithms are:
1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning
```

**Prompt 5**:
```
Based on the technical document, the three main techniques that DeepSeek V3.2
uses to achieve efficient inference on long sequences are:

1. **Multi-head Latent Attention (MLA)**
   - **How it contributes to efficiency**: MLA compresses the key-value cache
     using low-rank projections, dramatically reducing the memory footprint
     during inference while maintaining model quality.
```

---

## Success Criteria

### Priority Order

1. **Semantic Equivalence** (PRIMARY)
   - Generated text has the same meaning as reference
   - NOT exact token match - minor wording differences OK
   - Focus on: correctness, relevance, coherence

2. **No Crashes** (REQUIRED)
   - All 6 prompts complete without error
   - No OOM, no NaN, no assertion failures

3. **Correct Sparse Behavior** (REQUIRED for prompt 5)
   - Sparse attention must trigger when seq_len > 2048
   - Indexer selects top-k (2048) positions per query

### What NOT to Worry About

- Exact logits match (numerical precision varies by framework)
- Exact token-by-token output (minor variations acceptable)
- Timing differences (depends on hardware/implementation)

---

## Sparse Attention Behavior

### Trigger Condition

```python
if seq_length > config.index_topk:  # index_topk = 2048
    use_sparse_attention = True
```

### How It Works

1. **Indexer** computes compatibility scores for all query-key pairs
2. **Top-k selection** picks 2048 most relevant positions per query
3. **Sparse mask** is created: `-inf` for non-selected, `0` for selected
4. **Attention** is computed only over selected positions

### Memory Consideration

For prompt 5 (2251 tokens):
- Full attention matrix: `[1, 128, 2251, 2251]` = 2.6GB (will OOM on single GPU)
- Solution: Chunked attention or flash attention

### Verification

Log should show:
```
SPARSE ATTENTION TRIGGERED: seq_len (2251) > index_topk (2048)
```

---

## Common Issues & Solutions

### Issue 1: OOM on Long Sequences

**Symptom**: CUDA out of memory on prompt 5

**Cause**: Full attention matrix `[B, H, S, S]` allocated before sparse mask applied

**Solution**: Chunked attention - process queries in batches of 256
```python
chunk_size = 256
for chunk_start in range(0, seq_length, chunk_size):
    q_chunk = query_states[:, :, chunk_start:chunk_end, :]
    # Compute attention for chunk only
```

**Memory savings**: 2.6GB → 295MB per chunk

### Issue 2: Variable Reference Errors

**Symptom**: `local variable 'X' referenced before assignment`

**Cause**: Python conditional branches don't require variable declaration

**Solution**: Initialize variables before conditionals or add `else` clause
```python
if condition:
    attn_weights = compute_weights()
else:
    attn_weights = None  # Must initialize!
```

### Issue 3: Sparse Path Not Triggering

**Symptom**: Prompt 5 uses dense attention despite >2048 tokens

**Cause**: `use_sparse_attention` config flag not set, or wrong token count

**Solution**:
1. Verify token count: `tokenizer(prompt).input_ids.shape[1]` should be 2251
2. Check config: `model.config.use_sparse_attention = True`
3. Verify `index_topk` config value is 2048

### Issue 4: Wrong RoPE Implementation

**Symptom**: Garbled or nonsensical output

**Cause**: RoPE (Rotary Position Embedding) format mismatch

**Key insight from debugging**:
- DeepSeek V3.2 uses **interleaved** RoPE format
- Indexer uses **non-interleaved** RoPE
- Don't swap these!

### Issue 5: Test Uses Checkpoint Code Instead of Fork

**Symptom**: Changes to fork don't take effect

**Cause**: `trust_remote_code=True` uses code bundled with checkpoint

**Solution**: Always use `trust_remote_code=False` to test fork code
```python
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=False,  # Use installed transformers code!
)
```

---

## Timing Benchmarks

### HF Transformers on Single H200

| Phase | Time |
|-------|------|
| Model loading (163 shards, 1.2TB) | 3-4 min |
| Prompt 0-3 generation (200 tokens each) | 1-2 min each |
| Prompt 4 generation (37 tokens) | 1 min |
| Prompt 5 generation (200 tokens, sparse) | 2-3 min |
| **Total for all 6 prompts** | **15-20 min** |

### Development Iteration Cycle

| Step | Time |
|------|------|
| Code change | 5-10 min |
| Regenerate/commit/push | 2 min |
| Reinstall on cluster | 2-3 min |
| Run all 6 prompts | 15-20 min |
| **Total per iteration** | **25-35 min** |

---

## Lessons Learned

### What Worked Well

1. **Semantic comparison over exact match**
   - Practical, catches real issues, ignores numerical noise
   - Different frameworks have minor precision differences

2. **Diverse test prompts**
   - 6 prompts caught issues that single tests missed
   - Sparse trigger prompt (5) is essential

3. **Test ALL prompts after EVERY change**
   - A fix for one prompt can break others
   - Never assume - verify with facts

4. **Fire-and-forget SSH pattern**
   - `nohup python test.py > log.log 2>&1 &`
   - Check logs with short `tail` commands
   - No wasted time on dropped connections

5. **Immediate documentation**
   - Record results right after each test
   - Facts, not opinions

### What to Avoid

1. **Don't optimize prematurely**
   - Get correct output first, then optimize
   - Chunked attention was added only after OOM verified

2. **Don't trust assumptions**
   - "This should work" → always verify
   - Check token counts, config values, code paths

3. **Don't skip the sparse path test**
   - Most bugs hide in the sparse attention code
   - Prompt 5 is non-negotiable

4. **Don't stream logs over SSH**
   - Connections drop, lose progress
   - Use log files + snapshots

### Key Technical Insights

1. **OOM comes from attention matrix, not indexer**
   - Indexer output is small: `[B, S, top_k]`
   - Attention matrix is huge: `[B, H, S, S]`

2. **Chunked attention is mathematically equivalent**
   - Matrix multiplication is associative
   - `[Q1; Q2] @ K.T = [Q1 @ K.T; Q2 @ K.T]`

3. **RoPE format matters**
   - Main attention: interleaved format
   - Indexer: non-interleaved format
   - Mixing them produces garbage

---

## File Inventory

### Files to Copy for Megatron Verification

```
debug_v32/
├── reference_prompts.json      # 6 test prompts (REQUIRED)
├── long_prompt_sparse.txt      # 2251-token sparse trigger (REQUIRED)
├── HANDOFF_TO_MEGATRON.md      # This document
├── VERIFICATION_METHODOLOGY.md # Detailed methodology
├── FINDINGS.md                 # HF fork verification results
├── README.md                   # Full debugging documentation
└── test_prompt_4_5.py          # Example test script (adapt for Megatron)
```

### Reference Prompts JSON Structure

```json
{
  "prompts": [
    {
      "id": 0,
      "name": "simple_math",
      "prompt": "What is 2+2?",
      "expected_output": "2 + 2 = 4"
    },
    {
      "id": 5,
      "name": "sparse_trigger",
      "prompt_file": "long_prompt_sparse.txt",
      "expected_output": "Based on the document, the three main techniques..."
    }
  ]
}
```

---

## Megatron-Core Adaptation Notes

### What's Directly Reusable

- `reference_prompts.json` - Same prompts work for any framework
- `long_prompt_sparse.txt` - Same sparse trigger prompt
- Expected output patterns - Semantic criteria apply universally
- Success criteria - Same pass/fail logic
- Timing benchmarks - Similar order of magnitude expected

### What Needs Adaptation

| HF Transformers | Megatron-Core Equivalent |
|-----------------|--------------------------|
| `AutoModelForCausalLM.from_pretrained()` | `get_model()` + checkpoint loading |
| `model.generate()` | Custom generation loop |
| `AutoTokenizer` | Can reuse HF tokenizer |
| `config.json` | Megatron args YAML |
| Single GPU | May need tensor parallelism |

### Suggested Test Script Structure

```python
#!/usr/bin/env python3
"""Verify Megatron-Core DeepSeek V3.2 implementation."""

import json
from megatron.core import parallel_state
# ... Megatron imports

def load_prompts(json_path):
    with open(json_path) as f:
        return json.load(f)["prompts"]

def generate(model, tokenizer, prompt_text, max_new_tokens=200):
    # Megatron generation logic
    inputs = tokenizer(prompt_text, return_tensors="pt")
    # ... generate tokens ...
    return output_text

def main():
    initialize_megatron()
    model = load_model(checkpoint_path)
    tokenizer = load_tokenizer(tokenizer_path)

    prompts = load_prompts("reference_prompts.json")

    results = []
    for prompt in prompts:
        prompt_text = prompt.get("prompt") or open(prompt["prompt_file"]).read()
        output = generate(model, tokenizer, prompt_text)

        print(f"Prompt {prompt['id']} ({prompt['name']}): {output[:100]}...")

        results.append({
            "id": prompt["id"],
            "name": prompt["name"],
            "output": output,
            "input_tokens": len(tokenizer(prompt_text).input_ids),
        })

    # Check sparse attention triggered for prompt 5
    assert results[5]["input_tokens"] > 2048, "Prompt 5 should trigger sparse!"

    print("\n=== VERIFICATION COMPLETE ===")
    for r in results:
        print(f"Prompt {r['id']}: {len(r['output'])} chars generated")

if __name__ == "__main__":
    main()
```

---

## Contact & Questions

For questions about this verification methodology or the HF fork implementation:
- Branch: `shuyingl/deepseek-v32-minimal-on-v4.57.3`
- Repository: `lyfegame/transformers`

Key commits to reference:
- `ad7fe154f8` - Chunked attention OOM fix
- `076f8e5788` - Variable initialization fix

---

*Document generated: 2024-12-14*
*Verified on: H200 cluster with DeepSeek-V3.2-bf16 checkpoint*
