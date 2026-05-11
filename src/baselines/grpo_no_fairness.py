"""
GRPO (λ=0) Baseline for Fair-RLVR

Trains with GRPO using only the structural penalty and no fairness reward:

    R_total = 0 · R_fairness - P_structural  =  -P_structural

This is the critical ablation condition that isolates the contribution of the
fairness reward signal. If this baseline achieves a similar bias score to
Fair-RLVR, it would suggest that GRPO + structured reasoning alone drives
debiasing. If it stays near 0.5 (random errors), it confirms the explicit
fairness reward is the mechanism.

Slots into the lambda ablation table (Table 3) as the λ=0 row.

Usage:
    python -m src.baselines.grpo_no_fairness
    python -m src.baselines.grpo_no_fairness --output-dir results/grpo_lambda0
"""

import argparse
from pathlib import Path

from src.train import train
from src.evaluate import run_evaluation


def run_grpo_no_fairness(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    train_ratio: float = 0.9,
    learning_rate: float = 1e-5,
    num_train_steps: int = 3500,
    group_size: int = 2,
    batch_size: int = 8,
    gradient_accumulation: int = 2,
    max_new_tokens: int = 256,
    eval_batch_size: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    kl_coeff: float = 0.01,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True,
    output_dir: str = "results/grpo_no_fairness",
    save_steps: int = 500,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Train GRPO with λ=0 (no fairness reward) as an ablation baseline,
    then evaluate on the BBQ eval set.

    All hyperparameters are identical to Fair-RLVR except lambda_fair=0.0,
    ensuring the only variable between this run and the main model is the
    presence of the fairness reward signal.
    """
    print("\n" + "=" * 50)
    print("GRPO BASELINE: λ=0 (no fairness reward)")
    print("R_total = -P_structural only")
    print("=" * 50 + "\n")

    train(
        model_name=model_name,
        train_ratio=train_ratio,
        lambda_fair=0.0,          # ← the only difference from Fair-RLVR
        alpha_consistency=0.0,    # consistency reward off — keeps the ablation pure
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        group_size=group_size,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        max_new_tokens=max_new_tokens,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        kl_coeff=kl_coeff,
        use_4bit=use_4bit,
        gradient_checkpointing=gradient_checkpointing,
        output_dir=output_dir,
        save_steps=save_steps,
        seed=seed,
        dry_run=dry_run,
    )

    # ── Evaluate trained adapter ───────────────────────────
    # Load the saved adapter and run full BBQ evaluation so metrics are
    # available for the comparison table without a separate evaluate.py call.
    if not dry_run:
        checkpoint = str(Path(output_dir) / "final_adapter")
        print(f"\nRunning evaluation on {checkpoint}...")
        run_evaluation(
            checkpoint=checkpoint,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            batch_size=eval_batch_size,
            output_dir=str(Path(output_dir) / "eval"),
            seed=seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO λ=0 ablation baseline — no fairness reward signal"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Real GPU batch size for post-training eval")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing. Faster (~30%%) but uses more VRAM")
    parser.add_argument("--output-dir", type=str, default="results/grpo_no_fairness")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_grpo_no_fairness(
        model_name=args.model,
        train_ratio=args.train_ratio,
        learning_rate=args.lr,
        num_train_steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.eval_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        kl_coeff=args.kl_coeff,
        use_4bit=not args.no_4bit,
        gradient_checkpointing=not args.no_grad_checkpoint,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        seed=args.seed,
        dry_run=args.dry_run,
    )
