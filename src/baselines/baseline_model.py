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
    train_ratio: float = 0.9,
    n_eval: int = None,
    max_new_tokens: int = 512,
    batch_size: int = 64,
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
    # Decoder-only causal LMs must left-pad during generation so all prompts
    # end at the same position and generated tokens align across the batch.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    # ── Load data ──────────────────────────────────────────
    print("Loading BBQ dataset (10% eval split)...")
    splits = create_splits(train_ratio=train_ratio, seed=seed)
    eval_ds = splits["eval"]

    if n_eval is not None:
        eval_ds = eval_ds.select(range(min(n_eval, len(eval_ds))))

    eval_data = [eval_ds[i] for i in range(len(eval_ds))]
    print(f"Eval samples: {len(eval_data)}")

    # ── Run inference (real batched generation) ────────────
    print("Running zero-shot inference...")

    # Pre-format every prompt once so the batched loop is just tokenize+generate.
    prompt_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in eval_data
    ]

    generated_outputs = []
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch_prompts = prompt_texts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Left padding means every row's prompt ends at column input_len.
        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )
        generated_outputs.extend(decoded)

    predictions = [
        {
            "model_output": generated,
            "answer_label": example["answer_label"],
            "context_condition": example["context_condition"],
            "category": example["category"],
            "prompt": example["prompt"],
            "target_label": example.get("target_label"),
            "unknown_label": example.get("unknown_label", -1),
        }
        for example, generated in zip(eval_data, generated_outputs)
    ]

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
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Must match the ratio used during training to get the same eval split")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Max eval samples (default: full 10%% split)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results/baseline_model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — must match training seed to avoid eval/train overlap")
    args = parser.parse_args()

    run_zero_shot(
        model_name=args.model,
        train_ratio=args.train_ratio,
        n_eval=args.n_eval,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
