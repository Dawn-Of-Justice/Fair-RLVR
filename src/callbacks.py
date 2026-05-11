"""
Training Callbacks for Fair-RLVR

Logs per-step metrics during GRPO training:
- Reward components (fairness, consistency, structural penalty)
- Policy KL divergence
- Abstention rate
- CoT samples at checkpoints for qualitative analysis
- Weights & Biases integration (automatic when wandb.init() has been called)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

from transformers import TrainerCallback

from src.reward import extract_answer, extract_think, answer_to_index

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class FairRLVRCallback(TrainerCallback):
    """
    Callback that logs Fair-RLVR-specific metrics during GRPO training.

    Tracks:
    1. Reward component breakdown per step (fairness, consistency, structural penalty)
    2. Abstention rate (how often model selects the Unknown answer option)
    3. CoT samples at specified checkpoints for qualitative analysis
    4. All metrics logged to Weights & Biases when a W&B run is active
    """

    def __init__(
        self,
        output_dir: str = "results/training_logs",
        cot_checkpoint_steps: list[int] = None,
        n_cot_samples: int = 5,
        dynamics_logger=None,
    ):
        """
        Args:
            output_dir: directory to save logs and CoT samples
            cot_checkpoint_steps: training steps at which to save CoT samples
            n_cot_samples: number of CoT samples to save per checkpoint
            dynamics_logger: optional TrainingDynamicsLogger instance;
                if provided, log_generation_batch will call log_step() on it.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cot_checkpoint_steps = cot_checkpoint_steps or [100, 250, 500, 750, 1000]
        self.n_cot_samples = n_cot_samples
        self.dynamics_logger = dynamics_logger

        self.step_logs = []
        self.batch_logs = []
        self.cot_samples = {}

        # Synced to trainer.global_step in on_step_end; read by make_reward_fn
        # to stamp batch_logs with the correct step.
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics at the end of each training step."""
        self.current_step = state.global_step
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
            latest_batch = self.batch_logs[-1] if self.batch_logs else None
            base = (
                f"\n[Step {step}] "
                f"reward={(log_entry['reward_mean'] or 0.0):.3f} | "
                f"kl={(log_entry.get('kl_divergence') or 0.0):.4f} | "
                f"loss={(log_entry.get('loss') or 0.0):.4f}"
            )
            if latest_batch is not None:
                base += (
                    f"\n           "
                    f"acc={latest_batch.get('accuracy', 0):.3f} | "
                    f"r_fair={latest_batch.get('avg_r_fairness', 0):.3f} | "
                    f"r_cons={latest_batch.get('avg_r_consistency', 0):.3f} | "
                    f"p_struct={latest_batch.get('avg_p_structural', 0):.2f} | "
                    f"stereo_rate={latest_batch.get('stereotype_pick_rate_ambig', 0):.3f} | "
                    f"abstain={latest_batch.get('abstention_rate', 0):.3f} | "
                    f"sib_hit={latest_batch.get('sibling_hit_rate', 0):.3f}"
                )
            print(base)

    def on_save(self, args, state, control, **kwargs):
        self._save_logs(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._save_logs(state.global_step)
        print(f"\nTraining logs saved to {self.output_dir}")

    def _save_logs(self, step):
        """Write accumulated logs to disk."""
        log_path = self.output_dir / "step_logs.json"
        with open(log_path, "w") as f:
            json.dump(self.step_logs, f, indent=2)

        if self.batch_logs:
            batch_path = self.output_dir / "batch_logs.json"
            with open(batch_path, "w") as f:
                json.dump(self.batch_logs, f, indent=2)

        if self.cot_samples:
            with open(self.output_dir / "cot_samples.json", "w") as f:
                json.dump(self.cot_samples, f, indent=2)

    def log_generation_batch(
        self,
        step: int,
        completions: list[str],
        ground_truth_labels: list[int],
        precomputed_results: list,
        categories: list[str] = None,
        context_conditions: list[str] = None,
        unknown_labels: list[int] = None,
        target_labels: list[int] = None,
        lambda_fair: float = 0.5,
        sibling_hit_rate: float = 0.0,
    ):
        """
        Log detailed reward breakdown and CoT samples for a batch.

        Called automatically from make_reward_fn() in train.py — the reward
        function is the only point where GRPOTrainer exposes raw completions.

        `precomputed_results` is required. The precomputed dicts are the same ones
        the optimizer saw, so logged r_consistency reflects the actual alpha_consistency
        used during training (not a default-0 recomputation).

        Args:
            step: current training step (synced to trainer.global_step)
            completions: list of model output strings
            ground_truth_labels: list of BBQ ground truth labels
            precomputed_results: list of compute_reward() output dicts, one per completion
            categories: list of BBQ categories (optional)
            context_conditions: list of "ambig" or "disambig" (optional)
            unknown_labels: list of unknown option indices (0/1/2) (optional)
            target_labels: list of BBQ stereotype-aligned answer indices (optional)
            lambda_fair: fairness reward weight — stamped in per-step summary
            sibling_hit_rate: fraction of completions that had an in-batch sibling
        """
        if len(precomputed_results) != len(completions):
            raise ValueError(
                f"precomputed_results length {len(precomputed_results)} != "
                f"completions length {len(completions)}"
            )
        batch_stats = {
            "r_fairness": [],
            "r_consistency": [],
            "p_structural": [],
            "r_total": [],
            "abstained": 0,
            "total": 0,
            "correct": 0,
            "total_errors": 0,
            "format_failures": 0,
            "stereotype_picks_ambig": 0,
            "n_ambig": 0,
            "n_disambig": 0,
        }

        cot_examples = []

        for i, (completion, label) in enumerate(zip(completions, ground_truth_labels)):
            cond = context_conditions[i] if context_conditions is not None else None
            tgt = target_labels[i] if target_labels is not None else None
            result = precomputed_results[i]
            if result is None:
                raise ValueError(
                    f"precomputed_results[{i}] is None for label={label}; "
                    f"train.py should filter unmapped prompts before calling."
                )
            batch_stats["r_fairness"].append(result["r_fairness"])
            batch_stats["r_consistency"].append(result.get("r_consistency", 0.0))
            batch_stats["p_structural"].append(result["p_structural"])
            batch_stats["r_total"].append(result["r_total"])
            batch_stats["total"] += 1

            answer = extract_answer(completion)
            pred_idx = answer_to_index(answer)
            if pred_idx == -1:
                batch_stats["format_failures"] += 1
            elif pred_idx == label:
                batch_stats["correct"] += 1
            else:
                batch_stats["total_errors"] += 1

            if cond == "ambig":
                batch_stats["n_ambig"] += 1
                if tgt is not None and tgt >= 0 and pred_idx == tgt:
                    batch_stats["stereotype_picks_ambig"] += 1
            elif cond == "disambig":
                batch_stats["n_disambig"] += 1

            unknown_idx = unknown_labels[i] if unknown_labels is not None else -1
            if unknown_idx != -1 and pred_idx == unknown_idx:
                batch_stats["abstained"] += 1

            if len(cot_examples) < self.n_cot_samples:
                think = extract_think(completion)
                cot_examples.append({
                    "step": step,
                    "category": categories[i] if categories else "unknown",
                    "condition": cond if cond else "unknown",
                    "think": think[:500] if think else "",
                    "answer": answer or "",
                    "correct_label": label,
                    "target_label": tgt if tgt is not None else -1,
                    "is_correct": pred_idx == label,
                    "is_format_failure": pred_idx == -1,
                    "is_stereotype_pick": (
                        cond == "ambig" and tgt is not None
                        and tgt >= 0 and pred_idx == tgt
                    ),
                    "r_fairness": result["r_fairness"],
                    "p_structural": result["p_structural"],
                    "reward": result["r_total"],
                })

        n = batch_stats["total"]
        n_ambig = batch_stats["n_ambig"]

        def _avg(key):
            return sum(batch_stats[key]) / n if n > 0 else 0.0

        step_summary = {
            "step": step,
            "lambda_fair": lambda_fair,
            "avg_r_total": _avg("r_total"),
            "avg_r_fairness": _avg("r_fairness"),
            "avg_r_consistency": _avg("r_consistency"),
            "avg_p_structural": _avg("p_structural"),
            "accuracy": batch_stats["correct"] / n if n > 0 else 0.0,
            "format_failure_rate": batch_stats["format_failures"] / n if n > 0 else 0.0,
            "abstention_rate": batch_stats["abstained"] / n if n > 0 else 0.0,
            "stereotype_pick_rate_ambig": (
                batch_stats["stereotype_picks_ambig"] / n_ambig if n_ambig > 0 else 0.0
            ),
            "sibling_hit_rate": sibling_hit_rate,
            "n_samples": n,
            "n_ambig": n_ambig,
            "n_disambig": batch_stats["n_disambig"],
        }
        self.batch_logs.append(step_summary)

        if _WANDB_AVAILABLE and _wandb.run is not None:
            _wandb.log({
                "train/accuracy":              step_summary["accuracy"],
                "train/avg_reward":            step_summary["avg_r_total"],
                "train/r_fairness":            step_summary["avg_r_fairness"],
                "train/r_consistency":         step_summary["avg_r_consistency"],
                "train/p_structural":          step_summary["avg_p_structural"],
                "train/stereotype_rate_ambig": step_summary["stereotype_pick_rate_ambig"],
                "train/abstention_rate":       step_summary["abstention_rate"],
                "train/sibling_hit_rate":      step_summary["sibling_hit_rate"],
                "train/format_failure_rate":   step_summary["format_failure_rate"],
            })

        if step in self.cot_checkpoint_steps:
            self.cot_samples.setdefault(str(step), []).extend(cot_examples)
            print(f"\n[Step {step}] Saved {len(cot_examples)} CoT samples "
                  f"(total at step: {len(self.cot_samples[str(step)])})")
            for ex in cot_examples[:2]:
                print(f"  [{ex['category']}] correct={ex['is_correct']} | "
                      f"think: {ex['think'][:100]}...")

        if self.dynamics_logger is not None:
            self.dynamics_logger.log_step(step, completions)

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
