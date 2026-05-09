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

from src.reward import extract_answer, extract_think, answer_to_index


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

        # Synced to trainer.global_step in on_step_end; read by make_reward_fn
        # to stamp batch_logs with the correct step rather than a separate counter.
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics at the end of each training step."""
        self.current_step = state.global_step
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

            # Print periodic summary including the latest reward-component breakdown
            # (from log_generation_batch) when one is available.
            if step % 50 == 0:
                latest_batch = self.batch_logs[-1] if self.batch_logs else None
                # TRL doesn't always populate reward/kl/loss on the first few
                # log_history entries; coalesce None to 0.0 so formatting
                # never raises on early steps.
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
                        f"r_bin={latest_batch.get('avg_r_binary', 0):+.2f} | "
                        f"p_struct={latest_batch.get('avg_p_structural', 0):.2f} | "
                        f"p_stereo={latest_batch.get('avg_p_stereotype', 0):.3f} | "
                        f"stereo_rate={latest_batch.get('stereotype_pick_rate_ambig', 0):.3f} | "
                        f"abstain={latest_batch.get('abstention_rate', 0):.3f}"
                    )
                print(base)

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
        precomputed_results: list,
        categories: list[str] = None,
        context_conditions: list[str] = None,
        unknown_labels: list[int] = None,
        target_labels: list[int] = None,
        lambda_fair: float = 0.5,
    ):
        """
        Log detailed reward breakdown and CoT samples for a batch.

        Called automatically from make_reward_fn() in train.py — the reward
        function is the only point where GRPOTrainer exposes raw completions.

        `precomputed_results` is required (not optional). The callback used to
        recompute compute_reward() locally for logging, but that path didn't
        have access to alpha_consistency or sibling-pairing context, so logged
        r_consistency was always 0 even when training used α>0. Forcing the
        caller to pass the actual training rewards keeps batch_logs in sync
        with what the optimizer is actually seeing.

        Args:
            step: current training step (synced to trainer.global_step)
            completions: list of model output strings
            ground_truth_labels: list of BBQ ground truth labels
            precomputed_results: list of compute_reward() output dicts from the
                training reward function, one per completion. Same length and
                ordering as `completions`.
            categories: list of BBQ categories (optional)
            context_conditions: list of "ambig" or "disambig" (optional)
            unknown_labels: list of unknown option indices (0/1/2) (optional)
            target_labels: list of BBQ stereotype-aligned answer indices (optional);
                accepted for API compatibility, not used in R_total
            lambda_fair: fairness reward weight — recorded in the per-step
                summary as a stamp of what was used at this step
        """
        if len(precomputed_results) != len(completions):
            raise ValueError(
                f"precomputed_results length {len(precomputed_results)} != "
                f"completions length {len(completions)}"
            )
        batch_stats = {
            # Binary fairness reward (0.0 or +1.0)
            "r_fairness": [],
            # Counterfactual-consistency bonus (0.0 or +1.0)
            "r_consistency": [],
            # Structural penalty (0.0 to 1.2)
            "p_structural": [],
            # Total reward = lambda_fair * r_fairness + alpha * r_consistency - p_structural
            "r_total": [],
            # Aggregate counts
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
            # `precomputed_results[i]` is the same dict the optimizer saw —
            # logged numbers are guaranteed to match the actual training reward.
            result = precomputed_results[i]
            if result is None:
                # Caller filtered to valid labels in train.py before calling
                # this; a None here would indicate an upstream filter mismatch.
                raise ValueError(
                    f"precomputed_results[{i}] is None for label={label}; "
                    f"train.py should filter unmapped prompts before calling."
                )
            batch_stats["r_fairness"].append(result["r_fairness"])
            batch_stats["r_consistency"].append(result.get("r_consistency", 0.0))
            batch_stats["p_structural"].append(result["p_structural"])
            batch_stats["r_total"].append(result["r_total"])
            batch_stats["total"] += 1

            # Check accuracy / format / errors
            answer = extract_answer(completion)
            pred_idx = answer_to_index(answer)
            if pred_idx == -1:
                batch_stats["format_failures"] += 1
            elif pred_idx == label:
                batch_stats["correct"] += 1
            else:
                batch_stats["total_errors"] += 1

            # Track per-condition counts and stereotype picks (only meaningful in ambig)
            if cond == "ambig":
                batch_stats["n_ambig"] += 1
                if tgt is not None and tgt >= 0 and pred_idx == tgt:
                    batch_stats["stereotype_picks_ambig"] += 1
            elif cond == "disambig":
                batch_stats["n_disambig"] += 1

            # Abstention: model picked the per-question "Unknown" option index.
            # Don't hardcode index 2 — Unknown can be (a), (b), or (c).
            unknown_idx = unknown_labels[i] if unknown_labels is not None else -1
            if unknown_idx != -1 and pred_idx == unknown_idx:
                batch_stats["abstained"] += 1

            # Collect CoT sample
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

        # Compute averages
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
            # Stereotype rate is conditioned on ambig only — diagnostic metric.
            # NaN-safe: 0.0 when no ambig in batch.
            "stereotype_pick_rate_ambig": (
                batch_stats["stereotype_picks_ambig"] / n_ambig if n_ambig > 0 else 0.0
            ),
            "n_samples": n,
            "n_ambig": n_ambig,
            "n_disambig": batch_stats["n_disambig"],
        }
        self.batch_logs.append(step_summary)

        # Save CoT samples at checkpoint steps.
        # With gradient accumulation, log_generation_batch fires multiple times
        # per training step. Extend the bucket instead of overwriting so we
        # keep samples from every micro-batch.
        if step in self.cot_checkpoint_steps:
            self.cot_samples.setdefault(str(step), []).extend(cot_examples)
            print(f"\n[Step {step}] Saved {len(cot_examples)} CoT samples "
                  f"(total at step: {len(self.cot_samples[str(step)])})")
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
