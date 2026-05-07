---
name: Fair-RLVR Experiment Design
description: Planned experiments, baselines, metrics, and evaluation protocol
type: project
---

# Experiment Design

## Models
- **Baseline 1 (zero-shot):** Qwen2.5-3B-Instruct, no training (`src/baselines/baseline_model.py`)
- **Baseline 2 (SFT):** Qwen2.5-3B-Instruct + supervised fine-tune on BBQ answers (`src/baselines/sft.py`)
- **Baseline 3 (GRPO λ=0):** Qwen2.5-3B-Instruct + GRPO with no fairness reward — `R_total = -P_structural` only (`src/baselines/grpo_no_fairness.py`)
- **Ours (Fair-RLVR):** Qwen2.5-3B-Instruct + GRPO + `R_total = 0.5·R_fairness - P_structural`

> ⚠️ "GRPO correctness-only" baseline was removed. It was replaced by the λ=0 ablation, which directly tests the same scientific question (does fairness reward matter?) without depending on the now-removed R_correctness component.

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

- Log reward per training step (handled by `FairRLVRCallback.on_step_end`)
- Log abstention rate and per-step reward breakdown (handled by `log_generation_batch`)
- Phase classification logged by `TrainingDynamicsLogger` → saved to `results/fair_rlvr/dynamics/phase_log.json`
- CoT samples saved at: 100, 250, 500, 750, 1000 steps (configurable via `cot_checkpoint_steps`)
- Checkpoint model weights saved every 500 steps (3500 total → 7 checkpoints)
- Look for Phase 4 "The Hacker" (answer inside `<think>`) and Phase 6 "Reintegrated Reasoning"

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

## Compute Budget
Training is now 3,500 steps (was ~1,000 in early plan). Estimate per run varies by hardware:

| Hardware | Est. time per run | Notes |
|---|---|---|
| H100 80GB | ~6–8h | Recommended; used for original results |
| A100 40GB | ~10–12h | Feasible |
| T4 16GB | ~30–40h | Too slow for full sweep; use dry-run to validate |

| Experiment | Runs | Est. total (H100) |
|---|---|---|
| Exp 1 (4 conditions) | 4 runs | ~28h |
| Exp 2 (lambda sweep) | 5 runs | ~35h |
| Exp 3 (dynamics) | Included in Exp 1 | — |
| Exp 4 (faithfulness) | 1h post-train | ~1h |
| Exp 5 (OOD) | 2h post-train | ~2h |
| Exp 6 (bias amplification) | Logged during training | — |
| MMLU + GSM8K | 1h post-train | ~2h |
| **Total** | | **~70h (H100)** |
