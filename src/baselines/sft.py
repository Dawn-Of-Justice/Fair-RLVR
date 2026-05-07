"""
SFT Baseline for Fair-RLVR

Fine-tunes Qwen2.5-3B-Instruct on BBQ question-answer pairs using
supervised learning. No RL, no chain-of-thought reasoning.
This is Baseline 2 in Experiment 1.

The model learns to map BBQ prompts → correct answer labels directly.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from src.data import create_splits, SYSTEM_PROMPT, label_to_option
from src.evaluate import evaluate_all


def build_sft_dataset(split_data, tokenizer) -> Dataset:
    """
    Build SFT dataset: input = BBQ prompt, target = correct answer option.

    Each training example is a chat with:
    - System: SYSTEM_PROMPT
    - User: BBQ prompt
    - Assistant: "<think>\n[answer explanation]\n</think>\n<answer>(x)</answer>"

    Stores prompt_text and full_text separately so the tokenize step can
    mask prompt tokens (set labels=-100) and compute loss only on the
    assistant response.
    """
    rows = []
    for example in split_data:
        correct_option = label_to_option(example["answer_label"])

        if example["context_condition"] == "ambig":
            assistant_response = (
                f"<think>\nThe context does not provide enough information "
                f"to determine the answer.\n</think>\n"
                f"<answer>{correct_option}</answer>"
            )
        else:
            assistant_response = (
                f"<think>\nBased on the information provided in the context, "
                f"the answer can be determined.\n</think>\n"
                f"<answer>{correct_option}</answer>"
            )

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_response}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        rows.append({"prompt_text": prompt_text, "full_text": full_text})

    return Dataset.from_list(rows)


def train_sft(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    train_ratio: float = 0.9,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 768,
    output_dir: str = "results/sft",
    device: str = "auto",
    seed: int = 42,
):
    """
    Train SFT baseline on BBQ.

    Args:
        model_name: HuggingFace model name
        train_ratio: fraction of BBQ used for training (default 0.9)
        epochs: training epochs
        lr: learning rate
        batch_size: per-device batch size
        gradient_accumulation: gradient accumulation steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        max_seq_length: max sequence length
        output_dir: directory to save model and results
        device: device map
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
        torch_dtype=torch.bfloat16,   # Match train.py (bfloat16, not float16)
        device_map=device,
        trust_remote_code=True,
    )

    # ── LoRA ───────────────────────────────────────────────
    # Target modules must match train.py exactly so the SFT baseline has the
    # same number of trainable parameters as Fair-RLVR (fair comparison).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load data ──────────────────────────────────────────
    print("Loading BBQ dataset (full 90/10 split)...")
    splits = create_splits(train_ratio=train_ratio, seed=seed)

    train_dataset = build_sft_dataset(splits["train"], tokenizer)
    print(f"Training samples: {len(train_dataset)}")

    # Tokenize: compute loss only on assistant tokens by masking prompt with -100
    def tokenize(example):
        prompt_ids = tokenizer(
            example["prompt_text"],
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]
        full_tokens = tokenizer(
            example["full_text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        labels = full_tokens["input_ids"].copy()
        # Mask all prompt tokens — loss is computed only on the assistant response
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * prompt_len
        # Also mask padding tokens
        pad_id = tokenizer.pad_token_id
        labels = [
            -100 if (tok == pad_id and i >= prompt_len) else lbl
            for i, (tok, lbl) in enumerate(zip(full_tokens["input_ids"], labels))
        ]
        full_tokens["labels"] = labels
        return full_tokens

    train_dataset = train_dataset.map(tokenize, remove_columns=["prompt_text", "full_text"])

    # ── Train ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting SFT training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(str(output_path / "adapter"))
    tokenizer.save_pretrained(str(output_path / "adapter"))
    print(f"Adapter saved to {output_path / 'adapter'}")

    # ── Evaluate ───────────────────────────────────────────
    print("\nRunning evaluation...")
    model.eval()

    eval_data = [splits["eval"][i] for i in range(len(splits["eval"]))]

    predictions = []
    for example in tqdm(eval_data, desc="Evaluating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

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

    results = evaluate_all(
        predictions,
        output_path=str(output_path / "metrics.json"),
    )

    with open(output_path / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return results, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT baseline on BBQ")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results/sft")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — must match training seed to avoid eval/train overlap")
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
