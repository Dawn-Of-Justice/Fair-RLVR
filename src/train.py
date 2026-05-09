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

from src.data import create_splits, SYSTEM_PROMPT
from src.reward import compute_reward, predicted_answer_text
from src.callbacks import FairRLVRCallback, TrainingDynamicsLogger


# ── Reward wrapper for GRPOTrainer ────────────────────────

def make_reward_fn(
    ground_truth_map: dict,
    lambda_fair: float = 0.5,
    alpha_consistency: float = 0.0,
    callback=None,
):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls: reward_fn(completions, prompts=prompts, **dataset_columns)
    Extra dataset columns (category, context_condition, unknown_label,
    template_family_key, ans0/ans1/ans2) are forwarded via kwargs when
    remove_unused_columns=False is set in GRPOConfig.

    Args:
        ground_truth_map: dict mapping prompt text → ground truth label index
        lambda_fair: fairness reward weight
        alpha_consistency: counterfactual-consistency bonus weight (0.0 disables)
        callback: optional FairRLVRCallback — if provided, log_generation_batch()
                  is called each step so phase tracking and CoT logs are populated
    """
    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [])
        categories = kwargs.get("category", [None] * len(completions))
        context_conditions = kwargs.get("context_condition", [None] * len(completions))
        unknown_labels = kwargs.get("unknown_label", [None] * len(completions))
        target_labels = kwargs.get("target_label", [None] * len(completions))
        family_keys = kwargs.get("template_family_key", [None] * len(completions))
        ans0 = kwargs.get("ans0", [None] * len(completions))
        ans1 = kwargs.get("ans1", [None] * len(completions))
        ans2 = kwargs.get("ans2", [None] * len(completions))

        # ── Pre-compute predicted answer text per completion (for sibling pairing) ──
        # Map family_key → list of (completion_idx, predicted_text). Predictions
        # from completions where the prompt is from the SAME template family but
        # a different demographic fill are the "siblings" used for consistency.
        family_predictions: dict = {}
        per_completion_text: list = []
        for i, completion in enumerate(completions):
            opts = (ans0[i], ans1[i], ans2[i])
            txt = predicted_answer_text(completion, opts) if all(o is not None for o in opts) else None
            per_completion_text.append(txt)
            key = family_keys[i] if i < len(family_keys) else None
            if alpha_consistency > 0 and key is not None:
                family_predictions.setdefault(key, []).append((i, txt, prompts[i] if i < len(prompts) else None))

        rewards = []
        labels = []
        for i, completion in enumerate(completions):
            prompt_text = prompts[i] if i < len(prompts) else ""
            label = ground_truth_map.get(prompt_text, -1)

            if label == -1:
                print(f"[WARNING] Ground truth not found for prompt (step {i}). Returning 0.")
                rewards.append(0.0)
            else:
                # Sibling answer texts: predictions in the SAME family but from a
                # DIFFERENT prompt (different demographic fill). Skip same-prompt
                # entries (those are GRPO's G group-mates, not counterfactuals).
                sibling_texts: list = []
                if alpha_consistency > 0:
                    key = family_keys[i] if i < len(family_keys) else None
                    if key is not None:
                        sibling_texts = [
                            t for (j, t, p) in family_predictions.get(key, [])
                            if j != i and p != prompt_text and t is not None
                        ]
                opts = (ans0[i], ans1[i], ans2[i])
                result = compute_reward(
                    text=completion,
                    ground_truth_label=label,
                    context_condition=context_conditions[i] if i < len(context_conditions) else None,
                    target_label=target_labels[i] if i < len(target_labels) else None,
                    lambda_fair=lambda_fair,
                    alpha_consistency=alpha_consistency,
                    options=opts if all(o is not None for o in opts) else None,
                    sibling_answer_texts=sibling_texts,
                )
                rewards.append(result["r_total"])
            labels.append(label)

        # Wire into callback for live phase tracking and CoT logging.
        # Use callback.current_step (set in on_step_end) so the step number
        # stays in sync with trainer.global_step rather than a separate counter.
        if callback is not None:
            step = callback.current_step
            valid = [(c, l, cat, cond, unk, tgt)
                     for c, l, cat, cond, unk, tgt
                     in zip(completions, labels, categories, context_conditions,
                            unknown_labels, target_labels)
                     if l != -1]
            if valid:
                v_completions, v_labels, v_cats, v_conds, v_unks, v_tgts = zip(*valid)
                callback.log_generation_batch(
                    step=step,
                    completions=list(v_completions),
                    ground_truth_labels=list(v_labels),
                    categories=list(v_cats),
                    context_conditions=list(v_conds),
                    unknown_labels=list(v_unks),
                    target_labels=list(v_tgts),
                    lambda_fair=lambda_fair,
                )

        return rewards

    return reward_fn


# ── Dataset formatting for GRPOTrainer ────────────────────

def build_grpo_dataset(split_data, tokenizer, sort_by_family: bool = True) -> tuple[Dataset, dict]:
    """
    Build a dataset for GRPOTrainer and a ground truth lookup map.

    GRPOTrainer expects a dataset with a "prompt" column containing
    either a string or list of chat messages.

    Each row also exposes:
      - template_family_key: f"{category}:{question_index}:{context_condition}"
        Used by the counterfactual-consistency reward to identify siblings
        (same template, different demographic fill) within a batch.
      - ans0/ans1/ans2: option text strings, used to map (a)/(b)/(c) →
        canonical answer text for cross-variant comparison.

    When sort_by_family=True (default), the dataset is reordered so that
    siblings end up in adjacent rows, maximizing the chance they appear in
    the same GRPO reward batch and the consistency bonus actually fires.

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

        category = example["category"]
        qidx = example.get("question_index", -1)
        condition = example["context_condition"]
        family_key = f"{category}:{qidx}:{condition}"

        records.append({
            "prompt": prompt_str,
            "category": category,
            "context_condition": condition,
            "unknown_label": example.get("unknown_label", -1),
            # target_label = BBQ stereotype-aligned answer index (or -1 if absent).
            # Kept for API compatibility with compute_reward(); not used in reward formula.
            "target_label": example.get("target_label", -1) if example.get("target_label") is not None else -1,
            # Counterfactual-consistency support
            "template_family_key": family_key,
            "ans0": example.get("ans0", "") or "",
            "ans1": example.get("ans1", "") or "",
            "ans2": example.get("ans2", "") or "",
        })

        ground_truth_map[prompt_str] = example["answer_label"]

    if sort_by_family:
        # Adjacency = sibling co-batching. Stable sort on the family key.
        records.sort(key=lambda r: r["template_family_key"])

    dataset = Dataset.from_list(records)
    return dataset, ground_truth_map


# ── Main training function ────────────────────────────────

def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    train_ratio: float = 0.9,
    lambda_fair: float = 0.5,
    alpha_consistency: float = 0.0,
    learning_rate: float = 1e-5,
    num_train_steps: int = 3500,
    group_size: int = 2,
    batch_size: int = 8,
    gradient_accumulation: int = 2,
    max_new_tokens: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
    kl_coeff: float = 0.01,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.20,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True,
    output_dir: str = "results/fair_rlvr",
    save_steps: int = 500,
    logging_steps: int = 10,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Train Fair-RLVR with GRPO.

    Args:
        model_name: HuggingFace model name
        train_ratio: fraction of full BBQ dataset used for training (default 0.9)
        lambda_fair: fairness reward weight
        alpha_consistency: counterfactual-consistency bonus weight
            (Ravulu et al. 2024 CDA, adapted to RLVR). Default 0.0 (disabled).
            Recommended: 0.25.
        learning_rate: learning rate
        num_train_steps: total training steps
        group_size: G — number of completions per prompt
        batch_size: per-device batch size
        gradient_accumulation: gradient accumulation steps
        max_new_tokens: max generated tokens
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        kl_coeff: KL divergence coefficient
        clip_ratio_high: DAPO high clip ratio
        clip_ratio_low: DAPO low clip ratio
        use_4bit: use 4-bit quantization
        gradient_checkpointing: trade compute for VRAM by recomputing activations
            during backward (default True). Set False for ~30% speedup when VRAM
            is not the bottleneck.
        output_dir: directory to save model and results
        save_steps: save checkpoint every N steps
        logging_steps: log every N steps
        seed: random seed
        dry_run: if True, run only 5 steps to test pipeline
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        train_ratio = 0.01   # ~584 samples — enough to verify pipeline
        num_train_steps = 5
        save_steps = 5
        group_size = 2
        print("\n" + "=" * 50)
        print("DRY RUN MODE — 5 steps, ~1% of dataset")
        print("=" * 50 + "\n")

    # ── Save config ───────────────────────────────────────
    config = {
        "model_name": model_name,
        "train_ratio": train_ratio,
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
        "gradient_checkpointing": gradient_checkpointing,
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
        "attn_implementation": "sdpa",
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
    print("\nLoading BBQ dataset (full 90/10 split)...")
    splits = create_splits(train_ratio=train_ratio, seed=seed)

    train_dataset, ground_truth_map = build_grpo_dataset(splits["train"], tokenizer)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples:     {len(splits['eval'])}")
    print(f"Ground truth map entries: {len(ground_truth_map)}")

    # ── Callbacks ─────────────────────────────────────────
    dynamics_logger = TrainingDynamicsLogger(
        output_dir=str(output_path / "dynamics"),
    )
    fair_callback = FairRLVRCallback(
        output_dir=str(output_path / "logs"),
        cot_checkpoint_steps=[100, 500, 1000, 1500, 2000, 2500, 3000, 3500],
        dynamics_logger=dynamics_logger,
    )

    # ── Reward function ───────────────────────────────────
    # Pass the callback so log_generation_batch() is called from inside the
    # reward function — the only place GRPOTrainer exposes raw completions.
    reward_fn = make_reward_fn(
        ground_truth_map=ground_truth_map,
        lambda_fair=lambda_fair,
        alpha_consistency=alpha_consistency,
        callback=fair_callback,
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
        # KL and DAPO asymmetric clipping
        # epsilon      = low clip ratio  (standard lower bound)
        # epsilon_high = high clip ratio (DAPO "Clip-Higher" strategy)
        # Requires TRL >= 0.16 (epsilon_high added in 0.16.0).
        beta=kl_coeff,
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
        report_to="none",
        remove_unused_columns=False,
        # Gradient checkpointing for memory efficiency (toggle via --no-grad-checkpoint)
        gradient_checkpointing=gradient_checkpointing,
        # Optimizer: paged_adamw_8bit stores optimizer states in 8-bit and
        # pages them to CPU when VRAM is tight — a good match for 4-bit weights.
        optim="paged_adamw_8bit",
        # Overlap data loading with GPU compute on cloud instances (0 = serial).
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # model_init_kwargs belongs on GRPOConfig (not GRPOTrainer) per TRL >= 0.16
        model_init_kwargs=model_kwargs,
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
    effective_batch = batch_size * gradient_accumulation
    print("\n" + "=" * 50)
    print(f"Starting Fair-RLVR Training")
    print(f"  Reward: R_total = λ · R_fairness + α · R_consistency - P_structural")
    print(f"  Model: {model_name}")
    print(f"  Steps: {num_train_steps}")
    print(f"  λ (fairness reward weight): {lambda_fair}")
    print(f"  α (consistency bonus weight): {alpha_consistency} "
          f"({'on' if alpha_consistency > 0 else 'off'})")
    print(f"  Group size: {group_size}")
    print(f"  Micro-batch: {batch_size}, grad-accum: {gradient_accumulation}, "
          f"effective batch: {effective_batch}")
    print(f"  4-bit: {use_4bit} | grad-checkpoint: {'on' if gradient_checkpointing else 'off'}")
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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing. Faster (~30%%) but uses more VRAM")

    # Data
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Fraction of BBQ used for training (default 0.9)")

    # Reward
    parser.add_argument("--lambda-fair", type=float, default=None)
    parser.add_argument("--alpha-consistency", type=float, default=None,
                        help="Counterfactual-consistency bonus weight (default 0.0 = off; "
                             "0.25 is the recommended on value)")

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Modes
    parser.add_argument("--dry-run", action="store_true", help="Run 5 steps to test pipeline")

    args = parser.parse_args()

    # Load config file if provided
    config_overrides = {}
    if args.config:
        config_overrides = load_config(args.config)
        print(f"Loaded config from {args.config}")

    def _resolve(cli_val, config_key, default_val):
        """Precedence: explicit CLI arg > config file value > hardcoded default."""
        if cli_val is not None:
            return cli_val
        if config_key in config_overrides:
            return config_overrides[config_key]
        return default_val

    # CLI args take precedence over config file, which takes precedence over defaults
    train_kwargs = {
        "model_name": _resolve(args.model, "model_name", "Qwen/Qwen2.5-3B-Instruct"),
        "train_ratio": _resolve(args.train_ratio, "train_ratio", 0.9),
        "lambda_fair": _resolve(args.lambda_fair, "lambda_fair", 0.5),
        "alpha_consistency": _resolve(args.alpha_consistency, "alpha_consistency", 0.0),
        "learning_rate": _resolve(args.lr, "learning_rate", 1e-5),
        "num_train_steps": _resolve(args.steps, "num_train_steps", 3500),
        "group_size": _resolve(args.group_size, "group_size", 2),
        "batch_size": _resolve(args.batch_size, "batch_size", 8),
        "gradient_accumulation": _resolve(args.grad_accum, "gradient_accumulation", 2),
        "max_new_tokens": _resolve(args.max_new_tokens, "max_new_tokens", 256),
        "lora_r": _resolve(args.lora_r, "lora_r", 16),
        "lora_alpha": _resolve(args.lora_alpha, "lora_alpha", 32),
        "kl_coeff": _resolve(args.kl_coeff, "kl_coeff", 0.01),
        "use_4bit": not args.no_4bit,
        "gradient_checkpointing": not args.no_grad_checkpoint,
        "output_dir": _resolve(args.output_dir, "output_dir", "results/fair_rlvr"),
        "save_steps": _resolve(args.save_steps, "save_steps", 500),
        "seed": _resolve(args.seed, "seed", 42),
        "dry_run": args.dry_run,
    }

    train(**train_kwargs)
