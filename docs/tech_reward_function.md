---
name: Fair-RLVR Reward Function Design
description: Detailed breakdown of the composite reward function — simplified to 2-component form
type: project
---

# Reward Function: R_total

**Current (implemented) equation:**
```
R_total = λ · R_fairness - P_structural
```

> ⚠️ This supersedes the earlier 4-component design. R_correctness and P_leak were removed after analysis. See rationale below.

## Components

### R_fairness
- Does the final `<answer>` match the BBQ ground truth label?
- BBQ labels: index 0, 1, or 2 (which of the three options is correct)
- Binary: +1 for match, 0 for mismatch
- λ controls the weight — default λ=0.5, ablated across {0.1, 0.3, 0.5, 0.7, 1.0}
- Implemented in `src/reward.py` → `reward_fairness()`

### P_structural (structural penalty)
Rule-based, checks three violations (each = 0.3 penalty, max 0.9):
1. Answer leaking into `<think>` block (model reveals answer in reasoning)
2. Reasoning too short (< 20 tokens in `<think>`)
3. Content exists outside `<think>` / `<answer>` tags

Based on: Tarek et al. "Reward Hacking Mitigation using Verifiable Composite Rewards" (2025)
Implemented in `src/reward.py` → `penalty_structural()`

### λ (lambda)
- Controls fairness signal strength
- Even λ=0.1 achieves strong debiasing (empirically from our results)
- λ=0 → pure structural training → this is the `grpo_no_fairness.py` ablation baseline

## Why R_correctness was Removed
- R_correctness rewarded correct format (+1) / penalized bad format (-1)
- P_structural already captures all format violations with finer granularity
- Once format is learned, R_correctness normalizes to zero across a GRPO group — it provides no within-group signal
- Removing it simplifies the equation without any empirical loss

## Why P_leak was Removed
- P_leak used Sentence-BERT cosine similarity between `<think>` and `<answer>` content
- BBQ answers are single-letter options: `(a)`, `(b)`, `(c)` — short and sparse
- The cosine similarity of a one-word answer to a paragraph of reasoning will never exceed τ=0.85 in practice
- P_leak would therefore never fire on this task → dead code, adds SBERT dependency for zero benefit

## Tuning Risks
- λ too high → over-abstention: model learns "Unknown" is always safe, ignores disambiguated questions
- λ too low → fairness signal drowned out, structural training dominates
- Both are covered by the λ ablation experiment

## Ablation Plan (λ=0 replaces R_correctness-only baseline)
| Condition | Equation | Purpose |
|---|---|---|
| λ=0 (`grpo_no_fairness.py`) | `R_total = -P_structural` | Does GRPO alone improve fairness? |
| λ=0.1 | `R_total = 0.1·R_fairness - P_structural` | Minimum fairness signal |
| λ=0.3 | `R_total = 0.3·R_fairness - P_structural` | Conservative fairness |
| λ=0.5 (**default**) | `R_total = 0.5·R_fairness - P_structural` | Main experiment |
| λ=0.7 | `R_total = 0.7·R_fairness - P_structural` | Aggressive fairness |
| λ=1.0 | `R_total = R_fairness - P_structural` | Maximum fairness signal |
