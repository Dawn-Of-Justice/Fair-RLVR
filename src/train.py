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
import random
import yaml
import torch
from collections import defaultdict
from pathlib import Path
from typing import Optional
from peft import LoraConfig, TaskType
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from src.data import create_splits, SYSTEM_PROMPT
from src.reward import compute_reward, predicted_answer_text
from src.callbacks import FairRLVRCallback, TrainingDynamicsLogger


# ── Sibling-aware sampler ─────────────────────────────────

class FamilyGroupedSampler(Sampler):
    """
    Shuffles the order of template families but keeps members of the same
    family consecutive. This guarantees that siblings (same BBQ template,
    different demographic fill) land in the same GRPOTrainer reward batch,
    so the counterfactual-consistency bonus can actually fire.

    Replaces the default RandomSampler, which scatters siblings across batches
    regardless of how the dataset is sorted beforehand.
    """

    def __init__(self, dataset_family_keys: list, seed: int = 42):
        families: dict = defaultdict(list)
        for idx, key in enumerate(dataset_family_keys):
            families[key].append(idx)
        rng = random.Random(seed)
        family_groups = list(families.values())
        rng.shuffle(family_groups)
        self._indices = [idx for group in family_groups for idx in group]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


class FairGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer that swaps in FamilyGroupedSampler so siblings always
    co-appear in the same reward batch. Only activated when the dataset
    has a template_family_key column and use_family_sampler=True.
    """

    def __init__(self, *args, use_family_sampler: bool = False, sampler_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_family_sampler = use_family_sampler
        self._sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if (
            self._use_family_sampler
            and dataset is not None
            and "template_family_key" in dataset.column_names
        ):
            return FamilyGroupedSampler(
                dataset_family_keys=dataset["template_family_key"],
                seed=self._sampler_seed,
            )
        return super()._get_train_sampler(train_dataset)


# ── Reward wrapper for GRPOTrainer ────────────────────────

def make_reward_fn(
    ground_truth_map: dict,
    lambda_fair: float = 0.5,
    alpha_consistency: float = 0.0,
    callback=None,
    profile: bool = False,
):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls: reward_fn(completions, prompts=prompts, **dataset_columns)
    Extra dataset columns (category, context_condition, unknown_label,
    template_family_key, ans0/ans1/ans2) are forwarded via kwargs when
    remove_unused_columns=False is set in GRPOConfig.

    Args:
        ground_truth_map: dict mapping formatted prompt text -> ground truth label index
        lambda_fair: fairness reward weight
        alpha_consistency: counterfactual-consistency bonus weight (0.0 disables)
        callback: optional FairRLVRCallback; if provided, log_generation_batch()
                  is called each step so phase tracking and CoT logs are populated
        profile: if True, print per-call timing of generation+reward phases.
    """
    import time
    timer_state = {"last_call_end": None, "call_count": 0}

    def reward_fn(completions, **kwargs):
        if profile:
            t_start = time.time()
            if timer_state["last_call_end"] is not None:
                gen_train_time = t_start - timer_state["last_call_end"]
                print(f"[profile] gen+train phase: {gen_train_time:.2f}s "
                      f"({len(completions)} completions, "
                      f"{gen_train_time/max(len(completions),1):.3f}s/completion)")
        prompts = kwargs.get("prompts", [])
        categories = kwargs.get("category", [None] * len(completions))
        context_conditions = kwargs.get("context_condition", [None] * len(completions))
        unknown_labels = kwargs.get("unknown_label", [None] * len(completions))
        target_labels = kwargs.get("target_label", [None] * len(completions))
        family_keys = kwargs.get("template_family_key", [None] * len(completions))
        ans0 = kwargs.get("ans0", [None] * len(completions))
        ans1 = kwargs.get("ans1", [None] * len(completions))
        ans2 = kwargs.get("ans2", [None] * len(completions))

        # Pre-compute predicted answer text per completion (for sibling pairing).
        # Map family_key -> list of (completion_idx, predicted_text, prompt).
        family_predictions: dict = {}
        per_completion_text: list = []
        for i, completion in enumerate(completions):
            opts = (ans0[i], ans1[i], ans2[i])
            txt = predicted_answer_text(completion, opts) if all(o is not None for o in opts) else None
            per_completion_text.append(txt)
            key = family_keys[i] if i < len(family_keys) else None
            if alpha_consistency > 0 and key is not None:
                family_predictions.setdefault(key, []).append(
                    (i, txt, prompts[i] if i < len(prompts) else None)
                )

        rewards = []
        labels = []
        results: list = []
        n_with_siblings = 0

        for i, completion in enumerate(completions):
            prompt_text = prompts[i] if i < len(prompts) else ""
            label = ground_truth_map.get(prompt_text, -1)

            if label == -1:
                print(f"[WARNING] Ground truth not found for prompt (step {i}). Returning 0.")
                rewards.append(0.0)
                results.append(None)
            else:
                sibling_texts: list = []
                if alpha_consistency > 0:
                    key = family_keys[i] if i < len(family_keys) else None
                    if key is not None:
                        sibling_texts = [
                            t for (j, t, p) in family_predictions.get(key, [])
                            if j != i and p != prompt_text and t is not None
                        ]
                if sibling_texts:
                    n_with_siblings += 1
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
                results.append(result)
            labels.append(label)

        if callback is not None:
            step = callback.current_step
            sibling_hit_rate = n_with_siblings / len(completions) if completions else 0.0
            valid = [
                (c, l, cat, cond, unk, tgt, r)
                for c, l, cat, cond, unk, tgt, r
                in zip(completions, labels, categories, context_conditions,
                       unknown_labels, target_labels, results)
                if l != -1 and r is not None
            ]
            if valid:
                (v_completions, v_labels, v_cats, v_conds,
                 v_unks, v_tgts, v_results) = zip(*valid)
                callback.log_generation_batch(
                    step=step,
                    completions=list(v_completions),
                    ground_truth_labels=list(v_labels),
                    categories=list(v_cats),
                    context_conditions=list(v_conds),
                    unknown_labels=list(v_unks),
                    target_labels=list(v_tgts),
                    lambda_fair=lambda_fair,
                    precomputed_results=list(v_results),
                    sibling_hit_rate=sibling_hit_rate,
                )

        if profile:
            t_end = time.time()
            reward_time = t_end - t_start
            timer_state["call_count"] += 1
            timer_state["last_call_end"] = t_end
            print(f"[profile] reward fn:       {reward_time:.3f}s "
                  f"(call #{timer_state['call_count']})")

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
        sort_by_family: if True, sort by template_family_key so sibling variants
            land in adjacent positions (needed for R_consistency)

    Each row also exposes:
      - template_family_key: f"{category}:{question_index}:{context_condition}"
        Used by the counterfactual-consistency reward to identify siblings
        (same template, different demographic fill) within a batch.
      - ans0/ans1/ans2: option text strings, used to map (a)/(b)/(c) ->
        canonical answer text for cross-variant comparison.

    Returns:
        (dataset, ground_truth_map, family_map) where:
        - ground_truth_map: formatted prompt -> answer label index
        - family_map: formatted prompt -> (category, question_index)
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

        category = example["category"]
        condition = example["context_condition"]
        family_key = example.get("template_family_key", f"{category}:{condition}")

        records.append({
            "prompt": prompt_str,
            "category": category,
            "context_condition": condition,
            "unknown_label": example.get("unknown_label", -1),
            "target_label": example.get("target_label", -1) if example.get("target_label") is not None else -1,
            "template_family_key": family_key,
            "ans0": example.get("ans0", "") or "",
            "ans1": example.get("ans1", "") or "",
            "ans2": example.get("ans2", "") or "",
        })

        ground_truth_map[prompt_str] = example["answer_label"]
        family_map[prompt_str] = (example["category"], example.get("question_index", -1))

    if sort_by_family:
        records.sort(key=lambda r: r["template_family_key"])

    dataset = Dataset.from_list(records)

    n_missing = sum(1 for r in records if ground_truth_map.get(r["prompt"], -1) == -1)
    if n_missing > 0:
        print(f"[WARNING] {n_missing} training prompts have no ground truth label.")

    return dataset, ground_truth_map, family_map


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
    max_new_tokens: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    kl_coeff: float = 0.01,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.20,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True,
    output_dir: str = "results/fair_rlvr",
    save_steps: int = 500,
    logging_steps: int = 10,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "fair-rlvr",
    wandb_run_name: Optional[str] = None,
    dry_run: bool = False,
    profile: bool = False,
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
        group_size: G — number of completions per prompt (default 2 for 3-choice MCQA)
        batch_size: per-device batch size
        gradient_accumulation: gradient accumulation steps
        max_new_tokens: max generated tokens
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        kl_coeff: KL divergence coefficient
        clip_ratio_high: DAPO high clip ratio (epsilon_high)
        clip_ratio_low: DAPO low clip ratio (epsilon)
        lr_scheduler_type: HuggingFace scheduler name (default: cosine)
        warmup_ratio: fraction of steps for LR warmup (default 0.05)
        use_4bit: use 4-bit quantization
        gradient_checkpointing: recompute activations during backward (saves VRAM)
        output_dir: directory to save model and results
        save_steps: save checkpoint every N steps
        logging_steps: log every N steps
        seed: random seed (must be 42 across all scripts)
        use_wandb: enable Weights & Biases logging (requires `pip install wandb`)
        wandb_project: W&B project name (default: "fair-rlvr")
        wandb_run_name: W&B run name; if None, W&B auto-generates one
        dry_run: if True, run only 5 steps to test pipeline
        profile: if True, print per-call timing inside reward function
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
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "use_4bit": use_4bit,
        "gradient_checkpointing": gradient_checkpointing,
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
    tokenizer.padding_side = "left"

    # ── Load model ────────────────────────────────────────
    # Pre-load so we are not dependent on TRL's model_init_kwargs (not present
    # in all TRL >=0.12 builds). With 4-bit we also call
    # prepare_model_for_kbit_training before handing off to TRL/PEFT.
    print(f"\nLoading model: {model_name}")
    model_kwargs: dict = {
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
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if use_4bit:
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=gradient_checkpointing
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
        train_ratio=train_ratio,
        seed=seed,
        sort_by_family=sort_by_family,
    )

    train_dataset, ground_truth_map, family_map = build_grpo_dataset(
        splits["train"],
        tokenizer,
        sort_by_family=sort_by_family,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples:     {len(splits['eval'])}")
    print(f"Ground truth map entries: {len(ground_truth_map)}")

    # ── Callbacks ─────────────────────────────────────────
    cot_steps = [
        int(num_train_steps * frac)
        for frac in [0.03, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0]
        if int(num_train_steps * frac) > 0
    ]
    dynamics_logger = TrainingDynamicsLogger(
        output_dir=str(output_path / "dynamics"),
        use_wandb=use_wandb,
    )
    fair_callback = FairRLVRCallback(
        output_dir=str(output_path / "logs"),
        cot_checkpoint_steps=cot_steps,
        dynamics_logger=dynamics_logger,
    )

    # ── Reward function ───────────────────────────────────
    reward_fn = make_reward_fn(
        ground_truth_map=ground_truth_map,
        lambda_fair=lambda_fair,
        alpha_consistency=alpha_consistency,
        callback=fair_callback,
        profile=profile,
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
        beta=kl_coeff,
        # DAPO asymmetric clipping (requires TRL >=0.12)
        epsilon=clip_ratio_low,
        epsilon_high=clip_ratio_high,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=max(1, int(num_train_steps * warmup_ratio)),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        temperature=0.7,
        top_p=0.9,
        bf16=True,
        seed=seed,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    # ── Initialize trainer ────────────────────────────────
    print("\nInitializing FairGRPOTrainer...")
    trainer = FairGRPOTrainer(
        model=base_model,
        args=training_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[fair_callback],
        use_family_sampler=(alpha_consistency > 0),
        sampler_seed=seed,
    )

    # ── Train ─────────────────────────────────────────────
    effective_batch = batch_size * gradient_accumulation
    print("\n" + "=" * 55)
    print(f"Starting Fair-RLVR Training")
    print(f"  Reward: R_total = λ · R_fairness + α · R_consistency - P_structural")
    print(f"  Model:          {model_name}")
    print(f"  Steps:          {num_train_steps}")
    print(f"  λ (fairness):   {lambda_fair}")
    print(f"  α (consistency):{alpha_consistency} ({'on' if alpha_consistency > 0 else 'off'})")
    print(f"  Group size G:   {group_size}")
    print(f"  Micro-batch: {batch_size}, grad-accum: {gradient_accumulation}, "
          f"effective batch: {effective_batch}")
    print(f"  Clip (low/high):{clip_ratio_low}/{clip_ratio_high}")
    print(f"  4-bit: {use_4bit} | grad-checkpoint: {'on' if gradient_checkpointing else 'off'}")
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

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing (~30%% faster but uses more VRAM)")

    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Fraction of BBQ used for training (default 0.9)")

    parser.add_argument("--lambda-fair", type=float, default=None)
    parser.add_argument("--alpha-consistency", type=float, default=None,
                        help="Counterfactual-consistency bonus weight (default 0.0 = off; "
                             "0.25 is the recommended on value)")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-scheduler", type=str, default=None,
                        help="HuggingFace scheduler type (default: cosine)")
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)

    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--wandb", action="store_true", dest="use_wandb",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: fair-rlvr)")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="W&B run name (default: auto-generated by W&B)")

    parser.add_argument("--dry-run", action="store_true", help="Run 5 steps to test pipeline")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-call timing of generation+train phase vs reward fn")

    args = parser.parse_args()

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
        "max_new_tokens": _resolve(args.max_new_tokens, "max_new_tokens", 512),
        "lora_r": _resolve(args.lora_r, "lora_r", 16),
        "lora_alpha": _resolve(args.lora_alpha, "lora_alpha", 32),
        "kl_coeff": _resolve(args.kl_coeff, "kl_coeff", 0.01),
        "lr_scheduler_type": _resolve(args.lr_scheduler, "lr_scheduler_type", "cosine"),
        "warmup_ratio": _resolve(args.warmup_ratio, "warmup_ratio", 0.05),
        # --no-4bit / --no-grad-checkpoint override config; otherwise read from config
        "use_4bit": False if args.no_4bit else config_overrides.get("use_4bit", True),
        "gradient_checkpointing": False if args.no_grad_checkpoint else config_overrides.get("gradient_checkpointing", True),
        "output_dir": _resolve(args.output_dir, "output_dir", "results/fair_rlvr"),
        "save_steps": _resolve(args.save_steps, "save_steps", 500),
        "seed": _resolve(args.seed, "seed", 42),
        "use_wandb": args.use_wandb or config_overrides.get("use_wandb", False),
        "wandb_project": _resolve(args.wandb_project, "wandb_project", "fair-rlvr"),
        "wandb_run_name": _resolve(args.wandb_run, "wandb_run_name", None),
        "dry_run": args.dry_run,
        "profile": args.profile,
    }

    train(**train_kwargs)
