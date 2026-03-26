"""
General Benchmarks for Fair-RLVR — Alignment Tax Check

Runs MMLU and GSM8K to verify that fairness training does not
degrade general knowledge or mathematical reasoning.

Usage:
    # Zero-shot (no adapter)
    python -m src.benchmarks --output-dir results/zero_shot

    # Fair-RLVR checkpoint
    python -m src.benchmarks --checkpoint results/fair_rlvr/final_adapter --output-dir results/fair_rlvr
"""

import argparse
import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, checkpoint=None, device="auto"):
    """Load base model, optionally with a LoRA adapter."""
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

    if checkpoint:
        from peft import PeftModel
        print(f"Loading adapter from: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── MMLU ──────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def eval_mmlu(model, tokenizer, n_samples=500, seed=42):
    """
    Evaluate on MMLU (5-shot format, multiple choice).

    Args:
        model: loaded model
        tokenizer: tokenizer
        n_samples: total samples to evaluate across all subjects
        seed: random seed

    Returns:
        dict with accuracy and per-subject breakdown
    """
    print("\n" + "=" * 50)
    print("EVALUATING MMLU")
    print("=" * 50)

    dataset = load_dataset("cais/mmlu", "all", split="test")
    # Shuffle and take n_samples
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    choices = ["A", "B", "C", "D"]
    correct = 0
    total = 0
    per_subject = {}

    for example in tqdm(dataset, desc="MMLU"):
        subject = example["subject"]
        question = example["question"]
        options = example["choices"]
        answer_idx = example["answer"]

        prompt = f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\n"
        prompt += f"Question: {question}\n"
        for i, opt in enumerate(options):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "\nAnswer:"

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = generate(model, tokenizer, formatted, max_new_tokens=16)

        # Extract answer letter
        predicted = None
        output_clean = output.strip().upper()
        for c in choices:
            if output_clean.startswith(c) or f"({c})" in output_clean or f" {c}." in output_clean or f" {c} " in output_clean:
                predicted = c
                break
        # Fallback: first letter that's A-D
        if predicted is None:
            for char in output_clean:
                if char in choices:
                    predicted = char
                    break

        correct_letter = choices[answer_idx]
        is_correct = predicted == correct_letter

        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Per-subject accuracy
    subject_acc = {}
    for subj, stats in per_subject.items():
        subject_acc[subj] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    print(f"\nMMLU Accuracy: {accuracy:.3f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_subject": subject_acc,
    }


# ── GSM8K ─────────────────────────────────────────────────

def extract_gsm8k_answer(text):
    """Extract the final numerical answer from GSM8K model output."""
    # Look for #### pattern (GSM8K standard)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Look for "the answer is X" pattern
    match = re.search(r"(?:the answer is|answer:?)\s*\$?(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Look for answer tags
    match = re.search(r"<answer>\s*(-?[\d,]+\.?\d*)\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_gsm8k_gold(answer_text):
    """Extract gold answer from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def eval_gsm8k(model, tokenizer, n_samples=500, seed=42):
    """
    Evaluate on GSM8K (grade school math).

    Args:
        model: loaded model
        tokenizer: tokenizer
        n_samples: number of samples
        seed: random seed

    Returns:
        dict with accuracy
    """
    print("\n" + "=" * 50)
    print("EVALUATING GSM8K")
    print("=" * 50)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="GSM8K"):
        question = example["question"]
        gold = extract_gsm8k_gold(example["answer"])

        prompt = (
            "Solve this math problem step by step. "
            "Put your final numerical answer after ####.\n\n"
            f"Question: {question}\n\nSolution:"
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = generate(model, tokenizer, formatted, max_new_tokens=512)

        predicted = extract_gsm8k_answer(output)

        if predicted is not None and gold is not None:
            try:
                if float(predicted) == float(gold):
                    correct += 1
            except ValueError:
                pass
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nGSM8K Accuracy: {accuracy:.3f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ── Main ──────────────────────────────────────────────────

def run_benchmarks(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    checkpoint=None,
    n_samples=500,
    output_dir="results/benchmarks",
    device="auto",
):
    """Run MMLU and GSM8K benchmarks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_name, checkpoint, device)

    mmlu = eval_mmlu(model, tokenizer, n_samples=n_samples)
    gsm8k = eval_gsm8k(model, tokenizer, n_samples=n_samples)

    results = {
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "summary": {
            "mmlu_accuracy": mmlu["accuracy"],
            "gsm8k_accuracy": gsm8k["accuracy"],
        },
    }

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"  MMLU:  {mmlu['accuracy']:.3f}")
    print(f"  GSM8K: {gsm8k['accuracy']:.3f}")
    print("=" * 50)

    # Save
    out_file = output_path / "benchmark_results.json"
    # Remove per_subject for cleaner output
    save_results = json.loads(json.dumps(results))
    with open(out_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU and GSM8K benchmarks")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA adapter (omit for zero-shot)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Samples per benchmark")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmarks(
        model_name=args.model,
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
    )
