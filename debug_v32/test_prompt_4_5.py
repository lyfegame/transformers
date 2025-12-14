#!/usr/bin/env python3
"""Test DeepSeek V3.2 specifically on prompts 4 (long_context) and 5 (sparse_trigger).

IMPORTANT: This script uses trust_remote_code=False to ensure we test the
installed transformers fork code, NOT the code bundled with the checkpoint.
"""
import os
import json
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/models-local/DeepSeek-V3.2-bf16")
    parser.add_argument("--reference-dir", default="/mnt/models-disk/official_tensors")
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--use-sparse", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)  # Match reference output length
    parser.add_argument("--prompt-ids", type=str, default="4,5", help="Comma-separated prompt IDs")
    args = parser.parse_args()

    prompt_ids = [int(x) for x in args.prompt_ids.split(",")]

    print("=" * 70)
    print(f"TEST PROMPTS {prompt_ids}")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use sparse: {args.use_sparse}")
    print(f"Max tokens: {args.max_tokens}")

    # Load prompts
    with open(args.prompts_json) as f:
        prompts_data = json.load(f)
    prompts = prompts_data["prompts"]

    test_prompts = [p for p in prompts if p["id"] in prompt_ids]
    print(f"Testing {len(test_prompts)} prompts: {[p['name'] for p in test_prompts]}\n")

    # Load model - CRITICAL: trust_remote_code=False to use transformers fork code!
    print("Loading model (trust_remote_code=False)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Verify we're using the right transformers
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"Transformers path: {transformers.__file__}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,  # USE FORK CODE, NOT CHECKPOINT CODE!
    )

    if args.use_sparse:
        model.config.use_sparse_attention = True

    model.eval()
    print(f"Model loaded. use_sparse_attention={getattr(model.config, 'use_sparse_attention', False)}\n")

    for prompt_info in test_prompts:
        prompt_id = prompt_info["id"]
        prompt_name = prompt_info["name"]
        prompt_dir = prompt_info["directory"]

        print("=" * 70)
        print(f"PROMPT {prompt_id}: {prompt_name}")
        print("=" * 70)

        # Get prompt text
        if "prompt_file" in prompt_info:
            prompt_file = os.path.join(args.reference_dir, prompt_dir, prompt_info["prompt_file"])
            print(f"Loading prompt from: {prompt_file}")
            with open(prompt_file) as f:
                prompt_text = f.read().strip()
        else:
            prompt_text = prompt_info["prompt"]

        print(f"Prompt text (first 200 chars): {prompt_text[:200]}...")
        print(f"Prompt text length: {len(prompt_text)} chars")

        # Tokenize
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        print(f"Input tokens: {input_len}")

        # Check if sparse attention should be triggered
        index_topk = getattr(model.config, 'index_topk', 2048)
        if input_len > index_topk:
            print(f"  -> SPARSE ATTENTION TRIGGERED: seq_len ({input_len}) > index_topk ({index_topk})")
        else:
            print(f"  -> Dense attention: seq_len ({input_len}) <= index_topk ({index_topk})")

        # Generate text
        print(f"\n--- Generation Test ({args.max_tokens} tokens) ---")
        try:
            with torch.no_grad():
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = gen_outputs[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  Generated tokens: {len(generated_ids)}")
            print(f"  Generated text:\n{generated_text[:500]}...")
            print(f"\n  Expected (first 300 chars): {prompt_info.get('expected_output', 'N/A')[:300]}...")

        except Exception as e:
            print(f"  ERROR during generation: {e}")
            import traceback
            traceback.print_exc()

        # Clear cache
        torch.cuda.empty_cache()
        print()

    print("\nDone!")

if __name__ == "__main__":
    main()
