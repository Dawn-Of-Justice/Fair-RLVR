---
name: Fair-RLVR IEEE Paper Project
description: Accepted IEEE paper on training reasoning models to be fair via verifiable reward signals
type: project
---

# Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals

**Status:** Implementation complete — all code written and bug-fixed. Ready for GPU training.
**Last updated:** 2026-05-07

## Core Idea
Apply RLVR (Reinforcement Learning from Verifiable Rewards) — the same method behind DeepSeek-R1's emergent reasoning — to fairness/bias. Use BBQ (Bias Benchmark for QA, 58K labeled examples) as the verifiable ground truth for fairness, replacing subjective RLHF human feedback.

## Reward Function (current, implemented)
```
R_total = λ · R_fairness + α · R_consistency - P_structural
```
- **R_fairness:** BBQ ground truth label match (+1 / 0)
- **R_consistency:** Counterfactual-consistency bonus (+1 / 0). Fires when the predicted answer TEXT (option content, not the (a)/(b)/(c) index) matches an in-batch sibling from the same BBQ template family but a different demographic fill. Off by default (α=0); set α=0.25 in `configs/fair_rlvr.yaml`.
- **P_structural:** Structural penalties for format violations (0.3 each): answer leaked in think, think too short (<20 tokens), content outside tags
- **λ:** Fairness signal weight (default 0.5, ablated over {0.1, 0.3, 0.5, 0.7, 1.0})
- **α:** Consistency bonus weight (default 0.0; 0.25 in `fair_rlvr.yaml`)

> Earlier design had 4 components (R_correctness, R_fairness, P_structural, P_leak). R_correctness was removed (redundant with P_structural; zeros out in GRPO group). P_leak (Sentence-BERT similarity) was removed (single-letter MCQA answers never exceed the cosine similarity threshold τ=0.85). R_consistency was later added based on Ravulu et al. (IEEE AIxDKE 2024) — their Counterfactual Data Augmentation idea adapted to RLVR.

## Technical Stack (implemented)
- **Model:** Qwen2.5-3B-Instruct
- **Training:** GRPO (G=2) + LoRA (r=16, α=32) via TRL's GRPOTrainer
- **Quantization:** 4-bit (bitsandbytes nf4) during training; float16 at eval
- **DAPO fixes:** Asymmetric clipping (ε_high=0.28, ε_low=0.20)
- **Dataset:** Full BBQ — 52,643 training samples (90%), 5,849 eval samples (10%), seed=42
- **Steps:** 3,500 training steps, checkpoint every 500 steps
- **Hardware target:** H100 80GB (original results); A100/T4 feasible with longer runtime

## Deliverables
- `src/train.py` — GRPO training loop
- `src/reward.py` — Composite reward function
- `src/evaluate.py` — Full evaluation pipeline
- `src/data.py` — BBQ loading with `unknown_label` field
- `src/callbacks.py` — Training dynamics (6-phase) + CoT logging
- `src/baselines/baseline_model.py` — Zero-shot baseline
- `src/baselines/sft.py` — SFT baseline
- `src/baselines/grpo_no_fairness.py` — λ=0 GRPO ablation
- `configs/` — YAML configs for all experiments
- Published IEEE paper (in progress)

## Key Problems Addressed
1. "Vibe-based" RLHF fragility — noisy human labels → model mimics fair persona not fair logic
2. Lobotomy vs intelligence tradeoff — over-correction leads to refusal bias
3. Lack of causal faithfulness — CoT may be post-hoc rationalization

## Key Claims to Prove
- Fairness is verifiable (BBQ provides ground truth)
- GRPO forces causal bottleneck: reasoning → answer, not answer → reasoning
- Fair logic generalizes to out-of-distribution prompts (unlike result-only training)
- No "alignment tax" — fairness improves reasoning quality, not reduces it

## Why:
Accepted IEEE paper. User needs to now implement the code and write the paper.

## How to apply:
This is the primary ongoing project. All code, experiments, and writing should be scoped to this research. User will share details one by one before proceeding.
