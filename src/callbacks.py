"""
Training Callbacks for Fair-RLVR

Logs per-step metrics during GRPO training:
- Reward components (correctness, fairness, structural penalty, leak penalty)
- Policy entropy
- Abstention rate
- CoT samples at checkpoints for qualitative analysis
"""

import json
import re
from pathlib import Path
from collections import defaultdict

from transformers import TrainerCallback

from src.reward import extract_answer, extract_think, answer_to_index, compute_reward


class FairRLVRCallback(TrainerCallback):
    """
    Callback that logs Fair-RLVR-specific metrics during GRPO training.

    Tracks:
    1. Reward component breakdown per step
    2. Abstention rate (how often model says "Unknown")
    3. Bias score trend over training
    4. CoT samples at specified checkpoints
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

        # Running logs.
        # step_logs: entries from on_step_end (TRL trainer log_history, lightweight).
        # batch_logs: entries from log_generation_batch (detailed reward breakdown).
        # Kept separate to avoid mixed schemas in the saved JSON.
        self.step_logs = []
        self.batch_logs = []
        self.cot_samples = {}

    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics at the end of each training step."""
        # TRL's GRPOTrainer logs reward stats in state.log_history
        if state.log_history:
            latest = state.log_history[-1]
            step = state.global_step

            log_entry = {
                "step": step,
                "loss": latest.get("loss"),
                "learning_rate": latest.get("learning_rate"),
                # TRL GRPO logs these automatically
                "reward_mean": latest.get("reward"),
                "reward_std": latest.get("reward_std"),
                "kl_divergence": latest.get("kl"),
            }
            self.step_logs.append(log_entry)

            # Print periodic summary
            if step % 50 == 0:
                print(f"\n[Step {step}] "
                      f"reward={log_entry['reward_mean']:.3f} | "
                      f"kl={log_entry.get('kl_divergence', 0):.4f} | "
                      f"loss={log_entry.get('loss', 0):.4f}")

    def on_save(self, args, state, control, **kwargs):
        """Save logs when a checkpoint is saved."""
        self._save_logs(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Save final logs at end of training."""
        self._save_logs(state.global_step)
        print(f"\nTraining logs saved to {self.output_dir}")

    def _save_logs(self, step):
        """Write accumulated logs to disk."""
        # Trainer-level step logs (lightweight, from log_history)
        log_path = self.output_dir / "step_logs.json"
        with open(log_path, "w") as f:
            json.dump(self.step_logs, f, indent=2)

        # Detailed reward-breakdown logs (from log_generation_batch)
        if self.batch_logs:
            batch_path = self.output_dir / "batch_logs.json"
            with open(batch_path, "w") as f:
                json.dump(self.batch_logs, f, indent=2)

        if self.cot_samples:
            cot_path = self.output_dir / "cot_samples.json"
            with open(cot_path, "w") as f:
                json.dump(self.cot_samples, f, indent=2)

    def log_generation_batch(
        self,
        step: int,
        completions: list[str],
        ground_truth_labels: list[int],
        categories: list[str] = None,
        context_conditions: list[str] = None,
    ):
        """
        Log detailed reward breakdown and CoT samples for a batch.
        Call this manually from the training loop after each generation.

        Args:
            step: current training step
            completions: list of model output strings
            ground_truth_labels: list of BBQ ground truth labels
            categories: list of BBQ categories (optional)
            context_conditions: list of "ambig" or "disambig" (optional)
        """
        batch_stats = {
            "r_fairness": [],
            "p_structural": [],
            "r_total": [],
            "abstained": 0,
            "total": 0,
            "correct": 0,
            "total_errors": 0,
        }

        cot_examples = []

        for i, (completion, label) in enumerate(zip(completions, ground_truth_labels)):
            result = compute_reward(completion, label)
            batch_stats["r_fairness"].append(result["r_fairness"])
            batch_stats["p_structural"].append(result["p_structural"])
            batch_stats["r_total"].append(result["r_total"])
            batch_stats["total"] += 1

            # Check accuracy
            answer = extract_answer(completion)
            pred_idx = answer_to_index(answer)
            if pred_idx == label:
                batch_stats["correct"] += 1
            elif pred_idx != -1:
                batch_stats["total_errors"] += 1

            # Check abstention (answer is "Unknown" type — usually option (c) in ambiguous).
            # We use answer_to_index rather than a raw string check to avoid false matches.
            if answer is not None and answer_to_index(answer) == 2:
                batch_stats["abstained"] += 1

            # Collect CoT sample
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

        # Compute averages
        n = batch_stats["total"]
        step_summary = {
            "step": step,
            "avg_r_total": sum(batch_stats["r_total"]) / n if n > 0 else 0,
            "avg_r_fairness": sum(batch_stats["r_fairness"]) / n if n > 0 else 0,
            "avg_p_structural": sum(batch_stats["p_structural"]) / n if n > 0 else 0,
            "accuracy": batch_stats["correct"] / n if n > 0 else 0,
            "abstention_rate": batch_stats["abstained"] / n if n > 0 else 0,
            "n_samples": n,
        }
        self.batch_logs.append(step_summary)

        # Save CoT samples at checkpoint steps
        if step in self.cot_checkpoint_steps:
            self.cot_samples[str(step)] = cot_examples
            print(f"\n[Step {step}] Saved {len(cot_examples)} CoT samples")
            for ex in cot_examples[:2]:
                print(f"  [{ex['category']}] correct={ex['is_correct']} | "
                      f"think: {ex['think'][:100]}...")

        # Update dynamics logger if one was provided
        if self.dynamics_logger is not None:
            self.dynamics_logger.log_step(step, completions)

        return step_summary


class TrainingDynamicsLogger:
    """
    Standalone logger for Experiment 3: Training Dynamics.

    Tracks the 6 evolutionary phases from Med-RLVR:
    1. Format Failure
    2. Verbose Formatter
    3. Concise Structurer
    4. The Hacker (answer in think)
    5. The Exploit (formatting tricks)
    6. Reintegrated Reasoning
    """

    def __init__(self, output_dir: str = "results/training_dynamics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_log = []

    def classify_phase(self, completions: list[str]) -> dict:
        """
        Classify the current batch into one of the 6 training phases.

        Args:
            completions: list of model output strings

        Returns:
            dict with phase classification and stats
        """
        stats = {
            "no_tags": 0,           # Phase 1: Format Failure
            "verbose_empty": 0,     # Phase 2: Verbose Formatter
            "concise_good": 0,      # Phase 3: Concise Structurer
            "answer_leaked": 0,     # Phase 4: The Hacker
            "format_exploit": 0,    # Phase 5: The Exploit
            "real_reasoning": 0,    # Phase 6: Reintegrated Reasoning
            "total": len(completions),
        }

        for comp in completions:
            think = extract_think(comp)
            answer = extract_answer(comp)

            # Phase 1: No tags at all
            if think is None and answer is None:
                stats["no_tags"] += 1
                continue

            # Phase 2: Tags present but think is verbose gibberish
            if think and len(think.split()) > 50 and answer is None:
                stats["verbose_empty"] += 1
                continue

            if think and answer:
                think_words = think.split()
                think_lower = think.lower()

                # Phase 4: Answer leaked into think block
                answer_lower = answer.lower()
                leak_patterns = ["the answer is", "i'll go with", "select ("]
                if any(p in think_lower for p in leak_patterns):
                    stats["answer_leaked"] += 1
                    continue

                # Phase 5: Very short think, just enough to pass format check
                if len(think_words) < 10 and len(think_words) > 0:
                    stats["format_exploit"] += 1
                    continue

                # Phase 3 vs 6: Check reasoning quality
                reasoning_indicators = [
                    "context", "information", "evidence", "mention",
                    "specify", "determine", "because", "therefore",
                    "however", "although", "does not", "cannot",
                ]
                has_reasoning = sum(1 for r in reasoning_indicators if r in think_lower)

                if has_reasoning >= 3:
                    stats["real_reasoning"] += 1  # Phase 6
                elif len(think_words) >= 10:
                    stats["concise_good"] += 1   # Phase 3
                else:
                    stats["format_exploit"] += 1  # Phase 5

            else:
                stats["no_tags"] += 1

        # Determine dominant phase
        phase_map = {
            "no_tags": 1,
            "verbose_empty": 2,
            "concise_good": 3,
            "answer_leaked": 4,
            "format_exploit": 5,
            "real_reasoning": 6,
        }
        dominant = max(phase_map.keys(), key=lambda k: stats[k])
        stats["dominant_phase"] = phase_map[dominant]
        stats["dominant_phase_name"] = {
            1: "Format Failure",
            2: "Verbose Formatter",
            3: "Concise Structurer",
            4: "The Hacker",
            5: "The Exploit",
            6: "Reintegrated Reasoning",
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

    def save(self):
        """Save phase log to disk."""
        path = self.output_dir / "phase_log.json"
        with open(path, "w") as f:
            json.dump(self.phase_log, f, indent=2)
        print(f"Phase log saved to {path}")
