"""
Fair-RLVR Training Script

GRPO training loop using TRL's GRPOTrainer with the composite reward function.
Trains Qwen2.5-3B-Instruct with LoRA on BBQ using fairness as a verifiable reward.

Reward formula: R_total = λ·R_fairness + α·R_consistency - P_structural

Usage:
    python -m src.train --config configs/fair_rlvr.yaml
    python -m src.train --lambda-fair 0.5 --steps 3500
"""

import argparse
import json
import yaml
import torch
from collections import defaultdict
from pathlib import Path
from typing import Optional
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from src.data import create_splits, format_bbq_prompt, SYSTEM_PROMPT
from src.reward import compute_reward, extract_answer, answer_to_index
from src.callbacks import FairRLVRCallback, TrainingDynamicsLogger


# ── Reward wrapper for GRPOTrainer ────────────────────────

def make_reward_fn(
    ground_truth_map: dict,
    lambda_fair: float = 0.5,
    alpha_consistency: float = 0.0,
    family_map: Optional[dict] = None,
):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls: reward_fn(completions, prompts=prompts)

    Args:
        ground_truth_map: dict mapping formatted prompt text → ground truth label index
        lambda_fair: fairness reward weight
        alpha_consistency: counterfactual consistency bonus weight (0 = off)
        family_map: dict mapping formatted prompt text → (category, question_index) for sibling lookup
    """
    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [])

        # Build a prompt→completion map for sibling lookup within this batch
        prompt_to_completion = {}
        prompt_to_family = {}
        for i, (completion, prompt_text) in enumerate(zip(completions, prompts)):
            prompt_to_completion[prompt_text] = completion
            if family_map:
                prompt_to_family[prompt_text] = family_map.get(prompt_text)

        # Group prompts by template family for consistency reward
        family_groups = defaultdict(list)
        if alpha_consistency > 0 and family_map:
            for prompt_text in prompts:
                fam = family_map.get(prompt_text)
                if fam is not None:
                    family_groups[fam].append(prompt_text)

        rewards = []
        for i, completion in enumerate(completions):
            prompt_text = prompts[i] if i < len(prompts) else ""
            label = ground_truth_map.get(prompt_text, -1)

            if label == -1:
                # No ground truth found — format-only fallback; log warning
                print(f"[WARNING] No ground truth for prompt (truncated): {prompt_text[:80]!r}")
                from src.reward import penalty_structural
                rewards.append(-penalty_structural(completion))
                continue

            # Gather sibling completions (same template family, different demographic fill)
            sibling_texts = []
            if alpha_consistency > 0 and family_map:
                fam = family_map.get(prompt_text)
                if fam is not None:
                    for sib_prompt in family_groups.get(fam, []):
                        if sib_prompt != prompt_text and sib_prompt in prompt_to_completion:
                            sibling_texts.append(prompt_to_completion[sib_prompt])

            result = compute_reward(
                text=completion,
                ground_truth_label=label,
                lambda_fair=lambda_fair,
                alpha_consistency=alpha_consistency,
                sibling_texts=sibling_texts,
            )
            rewards.append(result["r_total"])

        return rewards

    return reward_fn


# ── Dataset formatting for GRPOTrainer ────────────────────

def build_grpo_dataset(
    split_data,
    tokenizer,
    sort_by_family: bool = False,
) -> tuple[Dataset, dict, dict]:
    """
    Build a dataset for GRPOTrainer and ground truth / family lookup maps.

    Args:
        split_data: BBQ split (HuggingFace Dataset)
        tokenizer: tokenizer for applying chat template
        sort_by_family: if True, sort by question_index so sibling variants
            land in adjacent positions (needed for R_consistency)

    Returns:
        (dataset, ground_truth_map, family_map) where:
        - ground_truth_map: formatted prompt → answer label index
        - family_map: formatted prompt → (category, question_index)
    """
    records = []
    ground_truth_map = {}
    family_map = {}

    for example in split_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        records.append({
            "prompt": prompt_str,
            "category": example["category"],
            "context_condition": example["context_condition"],
        })

        ground_truth_map[prompt_str] = example["answer_label"]
        family_map[prompt_str] = (example["category"], example.get("question_index", -1))

    dataset = Dataset.from_list(records)

    # Validate coverage
    n_missing = sum(1 for r in records if ground_truth_map.get(r["prompt"], -1) == -1)
    if n_missing > 0:
        print(f"[WARNING] {n_missing} training prompts have no ground truth label.")

    return dataset, ground_truth_map, family_map


# ── Main training function ────────────────────────────────

def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    lambda_fair: float = 0.5,
    alpha_consistency: float = 0.0,
    learning_rate: float = 1e-5,
    num_train_steps: int = 3500,
    group_size: int = 2,
    batch_size: int = 8,
    gradient_accumulation: int = 2,
    max_new_tokens: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    kl_coeff: float = 0.01,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.20,
    use_4bit: bool = True,
    output_dir: str = "results/fair_rlvr",
    save_steps: int = 500,
    logging_steps: int = 10,
    seed: int = 42,
    dry_run: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "fair-rlvr",
    wandb_run_name: Optional[str] = None,
):
    """
    Train Fair-RLVR with GRPO.

    Args:
        model_name: HuggingFace model name
        n_train: training samples (None = full dataset, ~52,643)
        n_eval: eval samples per condition (None = full eval split)
        lambda_fair: fairness reward weight
        alpha_consistency: counterfactual consistency bonus weight (0 = off)
        learning_rate: learning rate
        num_train_steps: total training steps
        group_size: G — number of completions per prompt (default 2 for 3-choice MCQA)
        batch_size: per-device batch size
        gradient_accumulation: gradient accumulation steps
        max_new_tokens: max generated tokens
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        kl_coeff: KL divergence coefficient
        clip_ratio_high: DAPO high clip ratio (ε_high=0.28)
        clip_ratio_low: DAPO low clip ratio (ε_low=0.20)
        use_4bit: use 4-bit NF4 quantization
        output_dir: directory to save model and results
        save_steps: save checkpoint every N steps
        logging_steps: log every N steps
        seed: random seed (must be 42 across all scripts)
        dry_run: if True, run only 5 steps to test pipeline
        use_wandb: enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated from config if None)
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
        "alpha_consistency": alpha_consistency,
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

    # ── Weights & Biases ──────────────────────────────────
    if use_wandb:
        try:
            import wandb
            run_name = wandb_run_name or f"lambda{lambda_fair}_alpha{alpha_consistency}_G{group_size}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
                dir=str(output_path),
            )
            print(f"W&B run: {wandb.run.url}")
        except ImportError:
            print("[WARNING] wandb not installed — disabling W&B. Run: pip install wandb")
            use_wandb = False

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
    sort_by_family = alpha_consistency > 0
    splits = create_splits(
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
        sort_by_family=sort_by_family,
    )

    train_dataset, ground_truth_map, family_map = build_grpo_dataset(
        splits["train"],
        tokenizer,
        sort_by_family=sort_by_family,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Ground truth map entries: {len(ground_truth_map)}")

    # ── Reward function ───────────────────────────────────
    reward_fn = make_reward_fn(
        ground_truth_map=ground_truth_map,
        lambda_fair=lambda_fair,
        alpha_consistency=alpha_consistency,
        family_map=family_map if alpha_consistency > 0 else None,
    )

    # ── Callbacks ─────────────────────────────────────────
    # Scale checkpoint steps proportionally to num_train_steps
    cot_steps = [
        int(num_train_steps * frac)
        for frac in [0.03, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0]
        if int(num_train_steps * frac) > 0
    ]
    fair_callback = FairRLVRCallback(
        output_dir=str(output_path / "logs"),
        cot_checkpoint_steps=cot_steps,
        use_wandb=use_wandb,
    )
    dynamics_logger = TrainingDynamicsLogger(
        output_dir=str(output_path / "dynamics"),
        use_wandb=use_wandb,
    )

    # ── GRPO Config ───────────────────────────────────────
    # DAPO asymmetric clipping: epsilon_high > epsilon prevents entropy collapse
    # while still penalizing low-probability over-exploitation (DAPO, arXiv:2503.14476)
    training_config = GRPOConfig(
        output_dir=str(output_path / "checkpoints"),
        max_steps=num_train_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_generations=group_size,
        max_completion_length=max_new_tokens,
        # KL regularization
        beta=kl_coeff,
        # DAPO asymmetric clipping (requires TRL >=0.12)
        epsilon=clip_ratio_low,
        epsilon_high=clip_ratio_high,
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
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
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
    print("\n" + "=" * 55)
    print(f"Starting Fair-RLVR Training")
    print(f"  Model:          {model_name}")
    print(f"  Steps:          {num_train_steps}")
    print(f"  λ (fairness):   {lambda_fair}")
    print(f"  α (consistency):{alpha_consistency}")
    print(f"  Group size G:   {group_size}")
    print(f"  Clip (low/high):{clip_ratio_low}/{clip_ratio_high}")
    print(f"  4-bit:          {use_4bit}")
    print("=" * 55 + "\n")

    trainer.train()

    # ── Save final model ──────────────────────────────────
    final_path = output_path / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nFinal adapter saved to {final_path}")

    dynamics_logger.save()

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("Training complete!")
    print(f"Results: {output_path}")
    print("=" * 55)

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
    parser.add_argument("--n-train", type=int, default=None,
                        help="Training samples (default: full dataset ~52,643)")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Eval samples per condition (default: full eval split)")

    # Reward
    parser.add_argument("--lambda-fair", type=float, default=0.5)
    parser.add_argument("--alpha-consistency", type=float, default=0.0,
                        help="Counterfactual consistency bonus weight (0 = off)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--kl-coeff", type=float, default=0.01)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Output
    parser.add_argument("--output-dir", type=str, default="results/fair_rlvr")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Modes
    parser.add_argument("--dry-run", action="store_true", help="Run 5 steps to test pipeline")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="fair-rlvr")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # Load config file if provided
    config_overrides = {}
    if args.config:
        config_overrides = load_config(args.config)
        print(f"Loaded config from {args.config}")

    # CLI args override config file
    train_kwargs = {
        "model_name": config_overrides.get("model_name", args.model),
        "n_train": config_overrides.get("n_train", args.n_train),
        "n_eval": config_overrides.get("n_eval", args.n_eval),
        "lambda_fair": config_overrides.get("lambda_fair", args.lambda_fair),
        "alpha_consistency": config_overrides.get("alpha_consistency", args.alpha_consistency),
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
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }

    train(**train_kwargs)
