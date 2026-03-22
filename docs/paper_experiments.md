---
name: Fair-RLVR Experiment Design
description: Planned experiments, baselines, metrics, and evaluation protocol
type: project
---

# Experiment Design

## Models
- **Primary:** Qwen2.5-3B-Instruct (T4-feasible)
- **Baseline 1:** Qwen2.5-3B-Instruct (no training — zero-shot BBQ)
- **Baseline 2:** Qwen2.5-3B-Instruct + SFT on BBQ answers (no RL)
- **Baseline 3:** Qwen2.5-3B-Instruct + GRPO, correctness-only reward (no fairness signal)
- **Ours:** Qwen2.5-3B-Instruct + GRPO + full R_total

## Experiment 1: Main Result
**Question:** Does Fair-RLVR reduce bias without degrading reasoning?

| Condition | BBQ (Ambig) | BBQ (Disambig) | MMLU | GSM8K |
|---|---|---|---|---|
| Zero-shot baseline | | | | |
| SFT baseline | | | | |
| GRPO (correctness only) | | | | |
| Fair-RLVR (full) | | | | |

Goal: Fair-RLVR best on BBQ, competitive on MMLU/GSM8K

## Experiment 2: Lambda Ablation
**Question:** What λ weight balances fairness vs utility?

| λ | BBQ Acc | MMLU | Abstention Rate |
|---|---|---|---|
| 0.1 | | | |
| 0.3 | | | |
| 0.5 | | | |
| 0.7 | | | |
| 1.0 | | | |

Watch for: λ>0.7 → over-abstention; λ<0.3 → fairness signal lost

## Experiment 3: Training Dynamics
**Question:** Do Med-RLVR's 6 phases appear in fairness training too?

- Log reward per training step
- Log entropy of policy distribution
- Log abstention rate over training
- Sample CoT chains at checkpoints: 100, 250, 500, 1000 steps
- Look for Phase 4 "Hacker" and Phase 6 "Reintegrated" equivalents

## Experiment 4: Causal Faithfulness
**Question:** Is the CoT causally linked to the answer, or post-hoc?

Method: Interventional Sufficiency Test
1. Take trained model output: (context, CoT, answer)
2. Corrupt CoT by randomly permuting sentences
3. Feed corrupted CoT → measure if answer changes
4. If answer degrades significantly → CoT is causally real
5. Metric: Faithfulness Score = P(correct answer | real CoT) - P(correct answer | corrupted CoT)

Threshold: score > 0.15 = causally faithful (to be validated)

## Experiment 5: Generalization (OOD Fairness)
**Question:** Does Fair-RLVR generalize beyond BBQ training distribution?

Eval on:
- **WinoBias** (gender/coreference — different format from BBQ)
- **StereoSet** (association-level bias — different task type)
- **Novel demographic** from BBQ held-out categories

## Experiment 6: Small Model Bias Amplification Check
**Question:** Does improved reasoning increase bias (FairReason warning)?

- Track bias score (stereotype-consistent errors / total errors) every 100 steps
- If score *increases* at any point, document the λ range where this occurs
- This is a potential negative result worth reporting

## Stress Test: Trick Prompts
Design 20 adversarial prompts where result-only model would fail:
- Prompts with demographic keyword that maps to common stereotype
- Prompts with misleading context that implies stereotyped answer
- Prompts with double negatives ("which person is LESS likely to be X")

Fair-RLVR should handle these via reasoning chain; SFT baseline should fail.

## Compute Budget (T4, 16GB)
- ~4 hours per training run (1K steps, 3B model, 4-bit)
- 5 main runs (Exp 1) × 4h = 20h
- 5 lambda values (Exp 2) × 4h = 20h
- Total: ~50h compute (including re-runs)
- Feasible within 6-week window
