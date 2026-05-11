---
name: Fair-RLVR Architecture Decision Record
description: Finalized design decisions for the Fair-RLVR IEEE paper implementation
type: adr
status: Accepted
date: 2026-05-07
---

# ADR-001: Fair-RLVR — Finalized Architecture Decisions

**Status:** Accepted  
**Date:** 2026-05-07  
**Deciders:** Salo E S, Arjun G Ravi, Devanand A, Rini Susan V S

---

## Context

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) to fairness
alignment using BBQ as the automated verifier. This ADR documents the finalized design
decisions and rationale for each, following full implementation and pre-training review.

---

## Decision 1: Reward Function — Three-Component Formula

### Decision
```
R_total = λ · R_fairness + α · R_consistency - P_structural
```

### Options Considered

| Component | Keep? | Rationale |
|---|---|---|
| `R_fairness` | ✅ | Core signal — BBQ ground truth match |
| `R_consistency` | ✅ (α=0 by default) | Counterfactual-consistency bonus across in-batch demographic-swap siblings (Ravulu et al. 2024 CDA, RLVR-adapted). Compares predicted answer TEXT, not (a)/(b)/(c) index. Off in the lambda sweep so the only varying reward component is λ; on (α=0.25) in `fair_rlvr.yaml` for the main run. |
| `P_structural` | ✅ | Actively penalizes format violations; fires on short think, leaks, outside-tag content |
| `R_correctness` | ❌ | Redundant: format violations already covered by P_structural. Zeros out in GRPO group normalization once format is stable. |
| `P_leak` (Sentence-BERT cosine sim) | ❌ | Never activates: single-letter MCQA answers `(a)/(b)/(c)` have near-zero cosine similarity with multi-sentence reasoning chains. τ=0.85 is never reached. Including it would claim anti-hacking behavior that doesn't occur. |

### Consequences
- Max reward for λ=0.5: `0.5 × 1.0 = 0.5`
- Reward range: `[-1.2, 0.5]` (four simultaneous violations → -1.2)
- Paper is internally consistent: simplicity is a feature, not a limitation

---

## Decision 2: Dataset — Full BBQ (52,643 train / 5,849 eval)

### Decision
Use the full BBQ dataset with a 90/10 stratified split (seed=42). No category holdout
during training. OOD evaluation uses WinoBias, StereoSet, and intersectional BBQ
categories (`race_x_gender`, `race_x_ses`).

### Options Considered

| Option | Assessment |
|---|---|
| Sample ~1,000 examples (original plan) | Insufficient for 3,500-step training at G=16. At batch=8, G=16, steps cycle through training set ~7× — 1K samples would cause extreme overfitting. |
| Full 58,492 (no split) | No held-out eval — can't measure generalization within BBQ. |
| **Full 90/10 stratified split (chosen)** | ~52K training samples, ~5.8K eval. Stratified by category to maintain balance. Clean separation with seeded split to prevent any data leakage. |

### Critical Implementation Detail
**All scripts (train.py, evaluate.py, all baselines) must use the same `--seed` (default: 42).**
Mismatched seeds produce different train/eval splits → data leakage. This is enforced
by threading `seed` through `create_splits()` in every script.

---

## Decision 3: Training Algorithm — GRPO + DAPO Improvements

### Decision
GRPO (G=16) with DAPO asymmetric clipping (ε_low=0.20, ε_high=0.28) and LoRA (r=16, α=32).
4-bit quantization (bitsandbytes nf4) during training; float16 at eval.

### Options Considered

| Option | Complexity | VRAM | Fairness Fit |
|---|---|---|---|
| PPO | High (critic model) | High (~2×) | Overkill |
| SFT only | Low | Low | Memorizes answers, no causal bottleneck |
| **GRPO (chosen)** | Medium | ~50% vs PPO | Forces group-relative policy improvement |
| DAPO (on top of GRPO) | Low add-on | Neutral | Prevents entropy collapse — critical at 3500 steps |

### DAPO Techniques Applied

| Technique | Applied | Config |
|---|---|---|
| Clip-Higher (asymmetric ε) | ✅ | ε_low=0.20, ε_high=0.28 |
| Token-level policy loss | ✅ | GRPOConfig default |
| Dynamic Sampling (skip all-same-reward batches) | ❌ | Not implemented — asymmetric clipping handles entropy collapse |
| Entropy regularization bonus | ❌ | Not needed — dynamic sampling handles it |

### LoRA Target Modules (all attention + MLP projections)
```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```
**All baselines (SFT, GRPO λ=0) use the same targets** to ensure equal trainable
parameter counts for a fair comparison.

---

## Decision 4: Ablation Baselines

### Decision
Three baselines to isolate each contribution:

| Baseline | Script | What it isolates |
|---|---|---|
| Zero-shot | `baseline_model.py` | Qwen2.5-3B-Instruct out-of-the-box fairness |
| SFT | `sft.py` | Does supervised learning on BBQ labels help? |
| GRPO λ=0 | `grpo_no_fairness.py` | Does GRPO + structured reasoning alone drive debiasing, without the fairness reward? |

The λ=0 baseline is the critical ablation. If it approaches Fair-RLVR's bias score,
GRPO alone explains the result. If it stays near 0.5 (random errors), the explicit
fairness reward is the mechanism.

---

## Decision 5: Abstention Detection — Per-Question `unknown_label` Index

### Decision
The "Unknown / Can't determine" answer option is **not always option (c)**. Its position
(a), (b), or (c) varies per BBQ question. Abstention must be detected by index comparison
using `get_unknown_label()` which scans `ans0/ans1/ans2` text for indicator phrases.

### Bug Fixed
Early implementation used either:
- String heuristic (`"unknown" in answer_text`) — never fires since model outputs `(a)/(b)/(c)`
- Hardcoded `predicted_idx == 2` — wrong for questions where "Unknown" is option (a) or (b)

Both produce incorrect abstention rates. The `unknown_label` field is now set by
`data.py` for every BBQ example and threaded through `evaluate.py` and `callbacks.py`.

---

## Decision 6: Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Training steps | 3,500 | Med-RLVR used ~3K steps for 3B model; allows full 6-phase observation |
| Checkpoint cadence | Every 500 steps | 7 checkpoints + final; fine-grained enough for phase tracking |
| Group size G | 16 | Double the GRPO default; more diverse per-prompt samples → better advantage estimation |
| Batch size | 8 per device | With grad_accum=2, effective batch = 16 |
| Max tokens | 512 | Allows 20-50 sentence reasoning chains comfortably |
| KL coefficient | 0.01 | Conservative; prevents excessive policy drift |
| Learning rate | 1e-5 | Standard for LoRA on 3B models |
| CoT checkpoint steps | 100, 500, 1000, 1500, 2000, 2500, 3000, 3500 | Covers all 6 training phases |

---

## Decision 7: `log_generation_batch` — Wired into Reward Function

### Decision
`FairRLVRCallback.log_generation_batch()` is called automatically from `make_reward_fn()`
in `train.py`. The reward function is the only point in TRL's GRPOTrainer where raw
completions are accessible — TRL's callback interface does not expose completions in
`on_step_end`. The callback is injected into the reward function closure, and
`callback.current_step` (updated in `on_step_end` to match `trainer.global_step`) is
used to stamp batch log entries with the correct step number.

Step-level reward mean/std (from `state.log_history`) is captured automatically via
`on_step_end`. Per-batch CoT samples and phase classification are captured live during
training via the reward function hook.

---

## Issues Found and Fixed in Final Review (2026-05-07)

| # | Severity | File | Issue | Fix |
|---|---|---|---|---|
| 1 | CRITICAL | `configs/*.yaml` | Stale params `n_train`, `n_eval`, `tau` silently ignored by `train.py` | Removed; replaced with `train_ratio: 0.9` |
| 2 | CRITICAL | `configs/fair_rlvr.yaml` | `num_train_steps: 1000` (should be 3500) | Fixed to 3500 |
| 3 | CRITICAL | `configs/lambda_sweep.yaml` | `num_train_steps: 1000` | Fixed to 3500 |
| 4 | IMPORTANT | `configs/*.yaml` | `save_steps: 250` (docs/paper say 500) | Fixed to 500 |
| 5 | IMPORTANT | `src/train.py` | `cot_checkpoint_steps` stopped at step 1000; training runs 3500 steps | Extended to [100, 500, 1000, 1500, 2000, 2500, 3000, 3500] |
| 6 | IMPORTANT | `src/evaluate.py` | CLI `--n-eval` defaulted to 500; paper uses full eval set (~5849) | Changed default to None |
| 7 | IMPORTANT | `src/baselines/sft.py` | `torch.float16` instead of `bfloat16` (mismatch with train.py) | Fixed to `bfloat16` |
| 8 | IMPORTANT | `src/baselines/sft.py` | LoRA targets only 4 modules (vs 7 in train.py) — unfair parameter count | Added gate_proj, up_proj, down_proj |
| 9 | IMPORTANT | `main.tex` | Old 4-component reward equation; stale G=8, 1000 steps, 1000 samples | Updated equation, all values synced |
| 10 | LOW | `configs/dry_run.yaml` | Same stale params as above | Cleaned up |

---

## Pre-Training Checklist

- [ ] Run `python -m src.train --dry-run` and confirm no errors in 5 steps
- [ ] Run `python -m src.data` and confirm 52,643 train / 5,849 eval split with correct category distribution
- [ ] Run `python -m src.reward` and confirm all 7 test cases produce expected outputs
- [ ] Confirm GPU has ≥80GB VRAM (H100) or use `--batch-size 4 --grad-accum 4` for A100
- [ ] Push to GitHub before launching GPU job
- [ ] After training: `python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter` (full eval set)
- [ ] After training: `python -m src.evaluate --checkpoint ... --run-faithfulness` (Experiment 4)
- [ ] Run all baselines with `--seed 42` to guarantee same train/eval split
