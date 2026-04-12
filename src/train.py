"""
Fair-RLVR Training Script

GRPO training loop using TRL's GRPOTrainer with the composite reward function.
Trains Qwen2.5-3B-Instruct with LoRA on BBQ using fairness as a verifiable reward.

Usage:
    python -m src.train --config configs/fair_rlvr.yaml
    python -m src.train --lambda-fair 0.5 --steps 1000
"""

import argparse
import json
import yaml
import torch
from pathlib import Path
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from src.data import create_splits, format_bbq_prompt, SYSTEM_PROMPT
from src.reward import compute_reward, extract_answer, answer_to_index
from src.callbacks import FairRLVRCallback, TrainingDynamicsLogger


# ── Reward wrapper for GRPOTrainer ────────────────────────

def make_reward_fn(ground_truth_map: dict, lambda_fair: float = 0.5, tau: float = 0.85):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls: reward_fn(completions, prompts=prompts)
    where completions is a list of strings and prompts is a list of prompt strings.

    Args:
        ground_truth_map: dict mapping prompt text → ground truth label index
        lambda_fair: fairness reward weight
        tau: semantic leak threshold
    """
    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            # Look up ground truth for this prompt
            prompt_text = prompts[i] if i < len(prompts) else ""

            # Try to find the ground truth label
            label = ground_truth_map.get(prompt_text, -1)

            if label == -1:
                # Fallback: only score format correctness
                from src.reward import reward_correctness
                rewards.append(reward_correctness(completion))
            else:
                result = compute_reward(
                    text=completion,
                    ground_truth_label=label,
                    lambda_fair=lambda_fair,
                    tau=tau,
                )
                rewards.append(result["r_total"])

        return rewards

    return reward_fn


# ── Dataset formatting for GRPOTrainer ────────────────────

def build_grpo_dataset(split_data, tokenizer) -> tuple[Dataset, dict]:
    """
    Build a dataset for GRPOTrainer and a ground truth lookup map.

    GRPOTrainer expects a dataset with a "prompt" column containing
    either a string or list of chat messages.

    Returns:
        (dataset, ground_truth_map) where ground_truth_map maps
        the formatted prompt string → answer label index.
    """
    records = []
    ground_truth_map = {}

    for example in split_data:
        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]

        # Format as string for the lookup map
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        records.append({
            "prompt": prompt_str,
            "category": example["category"],
            "context_condition": example["context_condition"],
        })

        ground_truth_map[prompt_str] = example["answer_label"]

    dataset = Dataset.from_list(records)
    return dataset, ground_truth_map


# ── Main training function ────────────────────────────────

def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_train: int = 1000,
    n_eval: int = 500,
    lambda_fair: float = 0.5,
    tau: float = 0.85,
    learning_rate: float = 1e-5,
    num_train_steps: int = 1000,
    group_size: int = 16,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    max_new_tokens: int = 512,
    max_prompt_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    kl_coeff: float = 0.01,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.20,
    use_4bit: bool = True,
    output_dir: str = "results/fair_rlvr",
    save_steps: int = 250,
    logging_steps: int = 10,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Train Fair-RLVR with GRPO.

    Args:
        model_name: HuggingFace model name
        n_train: number of BBQ training samples
        n_eval: number of BBQ eval samples
        lambda_fair: fairness reward weight
        tau: semantic leak threshold
        learning_rate: learning rate
        num_train_steps: total training steps
        group_size: G — number of completions per prompt
        batch_size: per-device batch size
        gradient_accumulation: gradient accumulation steps
        max_new_tokens: max generated tokens
        max_prompt_length: max prompt tokens
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        kl_coeff: KL divergence coefficient
        clip_ratio_high: DAPO high clip ratio
        clip_ratio_low: DAPO low clip ratio
        use_4bit: use 4-bit quantization
        output_dir: directory to save model and results
        save_steps: save checkpoint every N steps
        logging_steps: log every N steps
        seed: random seed
        dry_run: if True, run only 5 steps to test pipeline
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        n_train = 20
        n_eval = 10
        num_train_steps = 5
        save_steps = 5
        group_size = 2
        print("\n" + "=" * 50)
        print("DRY RUN MODE — 5 steps, 20 samples")
        print("=" * 50 + "\n")

    # ── Save config ───────────────────────────────────────
    config = {
        "model_name": model_name,
        "n_train": n_train,
        "n_eval": n_eval,
        "lambda_fair": lambda_fair,
        "tau": tau,
        "learning_rate": learning_rate,
        "num_train_steps": num_train_steps,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "max_new_tokens": max_new_tokens,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "kl_coeff": kl_coeff,
        "clip_ratio_high": clip_ratio_high,
        "clip_ratio_low": clip_ratio_low,
        "use_4bit": use_4bit,
        "seed": seed,
        "dry_run": dry_run,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path / 'config.json'}")

    # ── Load tokenizer ────────────────────────────────────
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    # ── Load model ────────────────────────────────────────
    print(f"Loading model: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ── LoRA config ───────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ── Load data ─────────────────────────────────────────
    print("\nLoading BBQ dataset...")
    splits = create_splits(n_train=n_train, n_eval=n_eval, seed=seed)

    train_dataset, ground_truth_map = build_grpo_dataset(splits["train"], tokenizer)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Ground truth map entries: {len(ground_truth_map)}")

    # ── Reward function ───────────────────────────────────
    reward_fn = make_reward_fn(
        ground_truth_map=ground_truth_map,
        lambda_fair=lambda_fair,
        tau=tau,
    )

    # ── Callbacks ─────────────────────────────────────────
    fair_callback = FairRLVRCallback(
        output_dir=str(output_path / "logs"),
        cot_checkpoint_steps=[100, 250, 500, 750, 1000],
    )
    dynamics_logger = TrainingDynamicsLogger(
        output_dir=str(output_path / "dynamics"),
    )

    # ── GRPO Config ───────────────────────────────────────
    training_config = GRPOConfig(
        output_dir=str(output_path / "checkpoints"),
        max_steps=num_train_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_generations=group_size,
        max_completion_length=max_new_tokens,
        # KL and clipping
        beta=kl_coeff,
        # Logging and saving
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        # Generation
        temperature=0.7,
        top_p=0.9,
        # Training
        bf16=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
    )

    # ── Initialize trainer ────────────────────────────────
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_name,
        args=training_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[fair_callback],
    )

    # ── Train ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Starting Fair-RLVR Training")
    print(f"  Model: {model_name}")
    print(f"  Steps: {num_train_steps}")
    print(f"  λ (fairness): {lambda_fair}")
    print(f"  τ (leak threshold): {tau}")
    print(f"  Group size: {group_size}")
    print(f"  4-bit: {use_4bit}")
    print("=" * 50 + "\n")

    trainer.train()

    # ── Save final model ──────────────────────────────────
    final_path = output_path / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal adapter saved to {final_path}")

    # Save training dynamics
    dynamics_logger.save()

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Results: {output_path}")
    print("=" * 50)

    return trainer


# ── CLI ───────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fair-RLVR with GRPO")

    # Config file (overrides defaults)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # Data
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=500)

    # Reward
    parser.add_argument("--lambda-fair", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.85)

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--kl-coeff", type=float, default=0.01)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Output
    parser.add_argument("--output-dir", type=str, default="results/fair_rlvr")
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    # Modes
    parser.add_argument("--dry-run", action="store_true", help="Run 5 steps to test pipeline")

    args = parser.parse_args()

    # Load config file if provided
    config_overrides = {}
    if args.config:
        config_overrides = load_config(args.config)
        print(f"Loaded config from {args.config}")

    # CLI args take precedence over config file
    train_kwargs = {
        "model_name": config_overrides.get("model_name", args.model),
        "n_train": config_overrides.get("n_train", args.n_train),
        "n_eval": config_overrides.get("n_eval", args.n_eval),
        "lambda_fair": config_overrides.get("lambda_fair", args.lambda_fair),
        "tau": config_overrides.get("tau", args.tau),
        "learning_rate": config_overrides.get("learning_rate", args.lr),
        "num_train_steps": config_overrides.get("num_train_steps", args.steps),
        "group_size": config_overrides.get("group_size", args.group_size),
        "batch_size": config_overrides.get("batch_size", args.batch_size),
        "gradient_accumulation": config_overrides.get("gradient_accumulation", args.grad_accum),
        "lora_r": config_overrides.get("lora_r", args.lora_r),
        "lora_alpha": config_overrides.get("lora_alpha", args.lora_alpha),
        "kl_coeff": config_overrides.get("kl_coeff", args.kl_coeff),
        "use_4bit": not args.no_4bit,
        "output_dir": config_overrides.get("output_dir", args.output_dir),
        "save_steps": config_overrides.get("save_steps", args.save_steps),
        "seed": config_overrides.get("seed", args.seed),
        "dry_run": args.dry_run,
    }

    train(**train_kwargs)
