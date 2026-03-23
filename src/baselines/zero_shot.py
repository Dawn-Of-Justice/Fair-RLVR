"""
Zero-Shot Baseline for Fair-RLVR

Runs Qwen2.5-3B-Instruct on BBQ with no additional training.
This is Baseline 1 in Experiment 1.
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
    n_eval_ambig: int = 500,
    n_eval_disambig: int = 500,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    output_dir: str = "results/zero_shot",
    device: str = "auto",
):
    """
    Run zero-shot evaluation on BBQ.

    Args:
        model_name: HuggingFace model name
        n_eval_ambig: number of ambiguous eval samples
        n_eval_disambig: number of disambiguated eval samples
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
    )
    model.eval()

    # ── Load data ──────────────────────────────────────────
    print("Loading BBQ dataset...")
    splits = create_splits(n_train=100, n_eval=max(n_eval_ambig, n_eval_disambig))

    # Combine ambiguous and disambiguated eval sets
    eval_data = []
    for i in range(min(n_eval_ambig, len(splits["eval_ambiguous"]))):
        eval_data.append(splits["eval_ambiguous"][i])
    for i in range(min(n_eval_disambig, len(splits["eval_disambiguated"]))):
        eval_data.append(splits["eval_disambiguated"][i])

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
    parser.add_argument("--n-eval", type=int, default=500, help="Eval samples per condition")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/zero_shot")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_zero_shot(
        model_name=args.model,
        n_eval_ambig=args.n_eval,
        n_eval_disambig=args.n_eval,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )
