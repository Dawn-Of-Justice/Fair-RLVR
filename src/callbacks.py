"""
Training Callbacks for Fair-RLVR

Logs per-step metrics during GRPO training:
- Reward components (fairness, consistency, structural penalty)
- Policy KL divergence
- Abstention rate
- CoT samples at checkpoints for qualitative analysis
- Weights & Biases integration (optional)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

from transformers import TrainerCallback

from src.reward import extract_answer, extract_think, answer_to_index, compute_reward


class FairRLVRCallback(TrainerCallback):
    """
    Callback that logs Fair-RLVR-specific metrics during GRPO training.

    Tracks:
    1. Reward component breakdown per step (fairness, consistency, structural penalty)
    2. Abstention rate (how often model selects the Unknown answer option)
    3. CoT samples at specified checkpoints for qualitative analysis
    4. All metrics logged to Weights & Biases when use_wandb=True
    """

    def __init__(
        self,
        output_dir: str = "results/training_logs",
        cot_checkpoint_steps: list[int] = None,
        n_cot_samples: int = 5,
        use_wandb: bool = False,
    ):
        """
        Args:
            output_dir: directory to save logs and CoT samples
            cot_checkpoint_steps: training steps at which to save CoT samples
            n_cot_samples: number of CoT samples to save per checkpoint
            use_wandb: whether to log metrics to Weights & Biases
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cot_checkpoint_steps = cot_checkpoint_steps or [100, 250, 500, 750, 1000]
        self.n_cot_samples = n_cot_samples
        self.use_wandb = use_wandb

        self.step_logs = []
        self.cot_samples = {}

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("[WARNING] wandb not installed. Run: pip install wandb")
                self.use_wandb = False

    def on_step_end(self, args, state, control, **kwargs):
        """Log TRL-provided reward/loss metrics at the end of each training step."""
        if not state.log_history:
            return

        latest = state.log_history[-1]
        step = state.global_step

        log_entry = {
            "step": step,
            "loss": latest.get("loss"),
            "learning_rate": latest.get("learning_rate"),
            "reward_mean": latest.get("reward"),
            "reward_std": latest.get("reward_std"),
            "kl_divergence": latest.get("kl"),
        }
        self.step_logs.append(log_entry)

        if step % 50 == 0:
            print(f"\n[Step {step}] "
                  f"reward={log_entry.get('reward_mean') or 0:.3f} | "
                  f"kl={log_entry.get('kl_divergence') or 0:.4f} | "
                  f"loss={log_entry.get('loss') or 0:.4f}")

        if self.use_wandb:
            wandb_log = {k: v for k, v in log_entry.items() if v is not None and k != "step"}
            self._wandb.log(wandb_log, step=step)

    def on_save(self, args, state, control, **kwargs):
        self._save_logs(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._save_logs(state.global_step)
        print(f"\nTraining logs saved to {self.output_dir}")

    def _save_logs(self, step):
        with open(self.output_dir / "step_logs.json", "w") as f:
            json.dump(self.step_logs, f, indent=2)
        if self.cot_samples:
            with open(self.output_dir / "cot_samples.json", "w") as f:
                json.dump(self.cot_samples, f, indent=2)

    def log_generation_batch(
        self,
        step: int,
        completions: list[str],
        ground_truth_labels: list[int],
        categories: list[str] = None,
        context_conditions: list[str] = None,
        lambda_fair: float = 0.5,
        alpha_consistency: float = 0.0,
    ):
        """
        Log detailed reward breakdown and CoT samples for a generation batch.

        Call this from the reward function (inside make_reward_fn) after each
        batch — it is the only point where raw completions are accessible.

        Args:
            step: current training step
            completions: list of model output strings
            ground_truth_labels: list of BBQ ground truth labels
            categories: list of BBQ categories (optional)
            context_conditions: list of "ambig" or "disambig" (optional)
            lambda_fair: fairness reward weight (for reward computation)
            alpha_consistency: consistency bonus weight
        """
        r_fairness_vals = []
        r_consistency_vals = []
        p_structural_vals = []
        r_total_vals = []
        correct = 0
        abstained = 0

        cot_examples = []

        for i, (completion, label) in enumerate(zip(completions, ground_truth_labels)):
            result = compute_reward(completion, label, lambda_fair=lambda_fair, alpha_consistency=alpha_consistency)
            r_fairness_vals.append(result["r_fairness"])
            r_consistency_vals.append(result["r_consistency"])
            p_structural_vals.append(result["p_structural"])
            r_total_vals.append(result["r_total"])

            answer = extract_answer(completion)
            pred_idx = answer_to_index(answer)
            if pred_idx == label:
                correct += 1

            # Unknown answer detection (heuristic for logging only)
            if answer and answer.strip("()") == "c":
                abstained += 1

            if len(cot_examples) < self.n_cot_samples:
                think = extract_think(completion)
                cot_examples.append({
                    "step": step,
                    "category": categories[i] if categories else "unknown",
                    "condition": context_conditions[i] if context_conditions else "unknown",
                    "think": think[:500] if think else "",
                    "answer": answer or "",
                    "correct_label": label,
                    "is_correct": pred_idx == label,
                    "reward": result["r_total"],
                })

        n = len(completions)
        step_summary = {
            "step": step,
            "avg_r_total": sum(r_total_vals) / n if n > 0 else 0,
            "avg_r_fairness": sum(r_fairness_vals) / n if n > 0 else 0,
            "avg_r_consistency": sum(r_consistency_vals) / n if n > 0 else 0,
            "avg_p_structural": sum(p_structural_vals) / n if n > 0 else 0,
            "accuracy": correct / n if n > 0 else 0,
            "abstention_rate": abstained / n if n > 0 else 0,
            "n_samples": n,
        }
        self.step_logs.append(step_summary)

        if self.use_wandb:
            self._wandb.log(
                {
                    "train/avg_r_total": step_summary["avg_r_total"],
                    "train/avg_r_fairness": step_summary["avg_r_fairness"],
                    "train/avg_r_consistency": step_summary["avg_r_consistency"],
                    "train/avg_p_structural": step_summary["avg_p_structural"],
                    "train/batch_accuracy": step_summary["accuracy"],
                    "train/batch_abstention_rate": step_summary["abstention_rate"],
                },
                step=step,
            )

        if step in self.cot_checkpoint_steps:
            self.cot_samples[str(step)] = cot_examples
            print(f"\n[Step {step}] Saved {len(cot_examples)} CoT samples")
            for ex in cot_examples[:2]:
                print(f"  [{ex['category']}] correct={ex['is_correct']} | "
                      f"think: {ex['think'][:100]}...")

        return step_summary


class TrainingDynamicsLogger:
    """
    Standalone logger for Experiment 3: Training Dynamics.

    Tracks the 6 evolutionary phases from Med-RLVR (Zhang et al. 2025):
    1. Format Failure       — no <think>/<answer> tags
    2. Verbose Formatter    — tags present, verbose but no real reasoning
    3. Concise Structurer   — clean format, adequate reasoning length
    4. The Hacker           — answer leaked into <think> block
    5. The Exploit          — minimal think just to satisfy format check
    6. Reintegrated Reasoning — genuine in-tag reasoning with evidence
    """

    def __init__(
        self,
        output_dir: str = "results/training_dynamics",
        use_wandb: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_log = []
        self.use_wandb = use_wandb

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self.use_wandb = False

    def classify_phase(self, completions: list[str]) -> dict:
        """
        Classify the current batch into one of the 6 training phases.

        Returns:
            dict with per-phase counts and dominant phase.
        """
        stats = {
            "no_tags": 0,
            "verbose_empty": 0,
            "concise_good": 0,
            "answer_leaked": 0,
            "format_exploit": 0,
            "real_reasoning": 0,
            "total": len(completions),
        }

        for comp in completions:
            think = extract_think(comp)
            answer = extract_answer(comp)

            if think is None and answer is None:
                stats["no_tags"] += 1
                continue

            if think and len(think.split()) > 50 and answer is None:
                stats["verbose_empty"] += 1
                continue

            if think and answer:
                think_words = think.split()
                think_lower = think.lower()

                leak_patterns = ["the answer is", "i'll go with", "select ("]
                if any(p in think_lower for p in leak_patterns):
                    stats["answer_leaked"] += 1
                    continue

                if len(think_words) < 10:
                    stats["format_exploit"] += 1
                    continue

                reasoning_indicators = [
                    "context", "information", "evidence", "mention",
                    "specify", "determine", "because", "therefore",
                    "however", "although", "does not", "cannot",
                ]
                has_reasoning = sum(1 for r in reasoning_indicators if r in think_lower)

                if has_reasoning >= 3:
                    stats["real_reasoning"] += 1
                else:
                    stats["concise_good"] += 1
            else:
                stats["no_tags"] += 1

        phase_map = {
            "no_tags": 1, "verbose_empty": 2, "concise_good": 3,
            "answer_leaked": 4, "format_exploit": 5, "real_reasoning": 6,
        }
        dominant = max(phase_map.keys(), key=lambda k: stats[k])
        stats["dominant_phase"] = phase_map[dominant]
        stats["dominant_phase_name"] = {
            1: "Format Failure", 2: "Verbose Formatter", 3: "Concise Structurer",
            4: "The Hacker", 5: "The Exploit", 6: "Reintegrated Reasoning",
        }[phase_map[dominant]]

        return stats

    def log_step(self, step: int, completions: list[str]):
        """Log phase classification for a training step."""
        phase_stats = self.classify_phase(completions)
        phase_stats["step"] = step
        self.phase_log.append(phase_stats)

        if step % 100 == 0:
            print(f"[Step {step}] Phase: {phase_stats['dominant_phase_name']} "
                  f"(real_reasoning={phase_stats['real_reasoning']}/{phase_stats['total']})")

        if self.use_wandb:
            self._wandb.log(
                {
                    "dynamics/dominant_phase": phase_stats["dominant_phase"],
                    "dynamics/real_reasoning_frac": phase_stats["real_reasoning"] / max(phase_stats["total"], 1),
                    "dynamics/format_failure_frac": phase_stats["no_tags"] / max(phase_stats["total"], 1),
                    "dynamics/hacker_frac": phase_stats["answer_leaked"] / max(phase_stats["total"], 1),
                },
                step=step,
            )

    def save(self):
        path = self.output_dir / "phase_log.json"
        with open(path, "w") as f:
            json.dump(self.phase_log, f, indent=2)
        print(f"Phase log saved to {path}")
