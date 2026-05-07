"""
Baseline Model for Fair-RLVR

Runs Qwen2.5-3B-Instruct on BBQ with no additional training.
This is the zero-shot baseline (Baseline 1) in the ablation study.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import create_splits, format_bbq_prompt, SYSTEM_PROMPT
from src.evaluate import evaluate_all


def run_zero_shot(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_eval: int = None,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    output_dir: str = "results/baseline_model",
    device: str = "auto",
    seed: int = 42,
):
    """
    Run zero-shot evaluation on BBQ.

    Args:
        model_name: HuggingFace model name
        n_eval: max eval samples (None = full 10% split, ~5,849 samples)
        max_new_tokens: max tokens for generation
        batch_size: inference batch size
        output_dir: directory to save results
        device: device to use ("auto", "cuda", "cpu")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # ── Load data ──────────────────────────────────────────
    print("Loading BBQ dataset (10% eval split)...")
    splits = create_splits(train_ratio=0.9, seed=seed)
    eval_ds = splits["eval"]

    if n_eval is not None:
        eval_ds = eval_ds.select(range(min(n_eval, len(eval_ds))))

    eval_data = [eval_ds[i] for i in range(len(eval_ds))]
    print(f"Eval samples: {len(eval_data)}")

    # ── Run inference ──────────────────────────────────────
    print("Running zero-shot inference...")
    predictions = []

    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[i : i + batch_size]

        for example in batch:
            # Build chat messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ]

            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only generated tokens
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            predictions.append({
                "model_output": generated,
                "answer_label": example["answer_label"],
                "context_condition": example["context_condition"],
                "category": example["category"],
                "prompt": example["prompt"],
                "target_label": example.get("target_label"),
                "unknown_label": example.get("unknown_label", -1),
            })

    # ── Evaluate ───────────────────────────────────────────
    results = evaluate_all(
        predictions,
        output_path=str(output_path / "metrics.json"),
    )

    # Save raw predictions
    with open(output_path / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_path / 'predictions.json'}")

    # Save sample outputs for manual inspection
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS (first 5)")
    print("=" * 60)
    for pred in predictions[:5]:
        print(f"\n--- [{pred['category']}] [{pred['context_condition']}] ---")
        print(f"Prompt: {pred['prompt'][:200]}...")
        print(f"Model output: {pred['model_output'][:300]}")
        print(f"Correct answer: {pred['answer_label']}")
        print()

    return results, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot baseline on BBQ")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Max eval samples (default: full 10%% split)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/baseline_model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — must match training seed to avoid eval/train overlap")
    args = parser.parse_args()

    run_zero_shot(
        model_name=args.model,
        n_eval=args.n_eval,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
