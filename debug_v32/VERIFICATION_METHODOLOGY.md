# DeepSeek V3.2 Implementation Verification Methodology

## Overview

This document describes a systematic methodology for verifying that a new implementation of DeepSeek V3.2 produces semantically equivalent outputs to the official inference code. This methodology was developed and validated while debugging the HuggingFace transformers fork.

**Applicability**: This methodology can be applied to:
- HuggingFace transformers implementations
- Megatron-Core implementations
- vLLM/SGLang serving implementations
- Any other framework implementing DeepSeek V3.2

## Core Principles

### 1. Semantic Equivalence Over Exact Match

**Why**: Different frameworks have minor numerical differences (floating point order, kernel implementations). Exact logits matching is impractical and unnecessary.

**Approach**: Compare generated text for semantic meaning, not token-by-token match.

| Comparison Type | Use Case | Tolerance |
|-----------------|----------|-----------|
| Semantic | Primary success metric | Same meaning |
| Logits cosine | Debugging divergence | > 0.95 acceptable |
| Top-k tokens | Quick sanity check | Top-1 should match most of time |

### 2. Diverse Test Coverage

**6 reference prompts** covering all code paths:

| ID | Name | Input Tokens | Attention Type | Tests |
|----|------|--------------|----------------|-------|
| 0 | simple_math | 11 | Dense | Short prompt, factual answer |
| 1 | greeting | 10 | Dense | Conversational response |
| 2 | code_generation | 16 | Dense | Code output formatting |
| 3 | explanation | 13 | Dense | Long-form explanation |
| 4 | long_context | 189 | Dense | Reading comprehension |
| 5 | sparse_trigger | 2251 | **Sparse** | Sparse attention path (seq > 2048) |

**Critical**: Prompt 5 is essential - it's the only one that triggers sparse attention.

### 3. Incremental Debugging

After **every code change**:
1. Commit and push (creates audit trail)
2. Clean reinstall on test environment
3. Test ALL 6 prompts (not just the one you think you fixed)
4. Document results immediately

**Never assume a change works - verify with facts.**

### 4. Fire-and-Forget Execution

For cluster/remote testing, never rely on long SSH sessions:

```bash
# BAD: Will break when SSH drops
ssh cluster 'python long_test.py'

# GOOD: Fire-and-forget
ssh cluster 'nohup python -u test.py > /tmp/test.log 2>&1 &'
# Later: check results
ssh cluster 'tail -100 /tmp/test.log'
```

## Reference Data

### Prompts (reference_prompts.json)

```json
{
  "prompts": [
    {"id": 0, "name": "simple_math", "prompt": "What is 2+2?"},
    {"id": 1, "name": "greeting", "prompt": "Hello! How are you today?"},
    {"id": 2, "name": "code_generation", "prompt": "Write a Python function to check if a number is prime."},
    {"id": 3, "name": "explanation", "prompt": "Explain the theory of relativity in simple terms."},
    {"id": 4, "name": "long_context", "prompt": "[1150 char document about ML]... What are the three main categories?"},
    {"id": 5, "name": "sparse_trigger", "prompt_file": "long_prompt_sparse.txt"}
  ]
}
```

### Expected Outputs (Semantic)

| Prompt | Expected Output Pattern |
|--------|------------------------|
| 0 | Contains "4" as the answer |
| 1 | Friendly greeting response |
| 2 | Valid `is_prime(n)` function with edge cases |
| 3 | Explains space, time, gravity relationship |
| 4 | Lists: supervised, unsupervised, reinforcement learning |
| 5 | Discusses MLA, Lightning Indexer, efficient inference |

### Official Reference Tensors

Location: `/mnt/models-disk/official_tensors/prompt_N_*/`

Each directory contains:
- `layer_N_*.pt` - Intermediate activations for debugging
- `indexer_output.pt` - Sparse attention indices (for prompt 5)
- `final_logits.pt` - Output logits for comparison

## Timing Analysis

### Single Test Run (All 6 Prompts)

| Phase | Time | Notes |
|-------|------|-------|
| Model loading | 3-4 min | 163 shards, ~1.2TB BF16 model |
| Prompts 0-3 (short) | 4-8 min | ~1-2 min each for 200 tokens |
| Prompt 4 (medium) | 1-2 min | 189 input tokens, 37 output |
| Prompt 5 (sparse) | 2-3 min | 2251 input, 200 output, chunked attention |
| **Total** | **15-20 min** | |

### Development Iteration Cycle

| Step | Time | Notes |
|------|------|-------|
| Identify issue | Variable | Depends on debugging complexity |
| Fix code locally | 5-10 min | Edit modular file |
| Regenerate modeling file | 10 sec | `python utils/modular_model_converter.py` |
| Commit and push | 1 min | Git operations |
| Reinstall on cluster | 2-3 min | Clone + pip install |
| Run full test | 15-20 min | All 6 prompts |
| Check results | 1 min | Parse log file |
| **Total cycle** | **25-35 min** | Per fix iteration |

### Debugging Session Timeline (Actual)

This debugging session fixed 2 issues:

| Time | Activity | Result |
|------|----------|--------|
| T+0 | Initial test (before fixes) | 4/6 pass, OOM on prompt 5 |
| T+30min | Implement chunked attention | Code complete |
| T+35min | Regenerate + commit + push | Commit `ad7fe154f8` |
| T+55min | Full test on cluster | New error: `attn_weights` undefined |
| T+60min | Fix variable reference | Commit `076f8e5788` |
| T+80min | Full test on cluster | **6/6 pass** |

**Total debugging time**: ~80 minutes for 2 fixes

## Adapting for Megatron-Core

### What Can Be Reused (90%)

1. **Reference prompts** (`reference_prompts.json`) - Framework agnostic
2. **Expected outputs** - Same semantic criteria
3. **Test coverage strategy** - 6 prompts cover all paths
4. **Timing benchmarks** - Model loading may differ, generation similar
5. **Documentation templates** - README, results tables
6. **SSH patterns** - Fire-and-forget works universally

### What Needs Adaptation (10%)

| Component | HF Transformers | Megatron-Core |
|-----------|-----------------|---------------|
| Model loading | `AutoModelForCausalLM.from_pretrained()` | `get_model()` with tensor parallelism |
| Tokenizer | `AutoTokenizer` | Same (can reuse HF tokenizer) |
| Generation | `model.generate()` | Custom generation loop |
| Checkpoint | HF safetensors | Megatron distributed checkpoints |
| Config | `config.json` | Megatron args + yaml |

### Megatron-Core Test Script Template

```python
#!/usr/bin/env python3
"""Test DeepSeek V3.2 Megatron-Core implementation."""
import json
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
# ... Megatron imports

def test_prompt(model, tokenizer, prompt_text, max_new_tokens=200):
    """Generate and return output text."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    # Megatron generation logic here
    output_ids = generate(model, inputs, max_new_tokens)
    return tokenizer.decode(output_ids)

def main():
    # Initialize Megatron distributed
    initialize_megatron()

    # Load model (Megatron checkpoint format)
    model = load_megatron_model(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load reference prompts (SAME FILE)
    with open("reference_prompts.json") as f:
        prompts = json.load(f)["prompts"]

    # Test all 6 prompts
    for prompt in prompts:
        output = test_prompt(model, tokenizer, prompt["prompt"])
        print(f"Prompt {prompt['id']}: {output[:100]}...")
        # Compare semantically with expected output
```

### Verification Checklist for New Implementation

- [ ] All 6 prompts complete without error
- [ ] Prompt 0: Returns "4" or equivalent
- [ ] Prompt 1: Friendly greeting response
- [ ] Prompt 2: Valid is_prime function
- [ ] Prompt 3: Explains relativity concepts
- [ ] Prompt 4: Lists 3 ML categories correctly
- [ ] Prompt 5: Sparse attention triggered (seq > 2048)
- [ ] Prompt 5: Discusses MLA/efficient inference

## Lessons Learned

### What Worked Well

1. **Semantic comparison** - Practical, catches real issues, ignores noise
2. **Diverse prompts** - Caught issues that single prompts would miss
3. **Sparse trigger prompt** - Essential for validating indexer path
4. **Fire-and-forget SSH** - No wasted time on dropped connections
5. **Immediate documentation** - Never lost track of what was tested

### What Could Be Improved

1. **Automated semantic comparison** - Currently manual; could use LLM judge
2. **Parallel prompt testing** - Run prompts concurrently to reduce time
3. **CI integration** - Automate test runs on PR/commit
4. **Logits checkpointing** - Save intermediate states for faster debugging

### Common Pitfalls

1. **Not testing all prompts** - A fix for one prompt can break others
2. **Assuming sparse path works** - Must explicitly test with >2048 tokens
3. **Forgetting to reinstall** - Old code cached in site-packages
4. **Streaming SSH logs** - Connection drops, lose test progress
5. **Exact match expectations** - Minor numerical differences are normal

## Files Included

```
debug_v32/
├── README.md                    # Main debugging documentation
├── VERIFICATION_METHODOLOGY.md  # This document
├── H200_CLUSTER_GUIDE.md        # Cluster SSH patterns
├── reference_prompts.json       # 6 test prompts
├── long_prompt_sparse.txt       # 2251-token sparse trigger prompt
├── test_prompt_4_5.py           # Test runner script
└── profile_inference.py         # Timing/profiling script
```

## Conclusion

This methodology provides a systematic, reproducible approach to verifying DeepSeek V3.2 implementations. The key insight is that **semantic equivalence matters more than numerical precision** - if the model produces meaningful, correct outputs, minor floating-point differences are acceptable.

For Megatron-Core or other implementations, reuse the reference prompts and expected outputs directly. Only the model loading and generation code needs adaptation.

---

*Document version: 2024-12-14*
*Validated on: HuggingFace transformers fork (branch: shuyingl/deepseek-v32-minimal-on-v4.57.3)*
