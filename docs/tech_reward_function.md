---
name: Fair-RLVR Reward Function Design
description: Detailed breakdown of the composite reward function — three-component form
type: project
---

# Reward Function: R_total

**Current (implemented) equation:**
```
R_total = λ · R_fairness + α · R_consistency - P_structural
```

> ⚠️ This supersedes the earlier 4-component design. R_correctness and P_leak were removed after analysis. R_consistency was later added based on Ravulu et al. (IEEE AIxDKE 2024) — their Counterfactual Data Augmentation idea adapted to RLVR. See rationale below.

## Components

### R_fairness
- Does the final `<answer>` match the BBQ ground truth label?
- BBQ labels: index 0, 1, or 2 (which of the three options is correct)
- Binary: +1 for match, 0 for mismatch
- λ controls the weight — default λ=0.5, ablated across {0.1, 0.3, 0.5, 0.7, 1.0}
- Implemented in `src/reward.py` → `reward_fairness()`

### R_consistency (counterfactual-consistency bonus)
- Compares the predicted answer TEXT (option content like "the grandfather"), not the (a)/(b)/(c) index — because demographic-swap variants permute option order.
- Fires when ≥1 in-batch sibling from the same BBQ template family but a different demographic fill predicted the same answer text. Binary: +1 if a sibling agrees, 0 otherwise.
- Sibling co-batching is achieved by `build_grpo_dataset(sort_by_family=True)`, which sorts the dataset on `(category, question_index, context_condition)` so siblings end up adjacent and likely land in the same reward batch.
- α controls the weight — default α=0 (off, used in the lambda sweep to isolate λ); α=0.25 in `configs/fair_rlvr.yaml` for the main run.
- Based on: Ravulu et al. "Mitigating Bias in RLHF for Large Language Models" (IEEE AIxDKE 2024) — Counterfactual Data Augmentation. RLVR-adapted: instead of duplicating examples and adding a paired-prompt loss, we use BBQ's existing demographic-fill structure and reward semantic agreement across in-batch siblings.
- Implemented in `src/reward.py` → `reward_consistency()` and `predicted_answer_text()`.

### P_structural (structural penalty)
Rule-based, checks four violations (each = 0.3 penalty, max 1.2):
1. Reasoning too short or `<think>` tag missing (< 20 tokens)
2. Missing `<answer>` tag entirely — long think with no answer is not rewarded
3. Answer leaking into `<think>` block (model reveals answer in reasoning)
4. Content exists outside `<think>` / `<answer>` tags

Based on: Tarek et al. "Reward Hacking Mitigation using Verifiable Composite Rewards" (2025)
Implemented in `src/reward.py` → `penalty_structural()`

### λ (lambda)
- Controls fairness signal strength
- Even λ=0.1 achieves strong debiasing (empirically from our results)
- λ=0 → pure structural training → this is the `grpo_no_fairness.py` ablation baseline

### α (alpha)
- Controls counterfactual-consistency strength
- α=0 disables the consistency bonus entirely (preserved as the ablation default)
- α=0.25 is the on-value used in the main `fair_rlvr.yaml` run
- Reward range becomes `[−1.2, λ + α]` — at λ=0.5, α=0.25 the upper bound is 0.75

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

## Why R_consistency was Added
- Ravulu et al. 2024 showed CDA (counterfactual data augmentation + consistency reward) is one of the most effective bias-mitigation strategies layered on top of standard RLHF.
- BBQ already encodes counterfactual structure — same template, different demographic fill — that was previously unused at training time.
- The bonus is binary and sibling-conditional, so it only fires when a meaningful counterfactual is present in the batch; α=0 keeps the original two-term formula intact for the lambda sweep.

## Tuning Risks
- λ too high → over-abstention: model learns "Unknown" is always safe, ignores disambiguated questions
- λ too low → fairness signal drowned out, structural training dominates
- α too high → model can hack consistency by always picking the same option regardless of evidence (consistency without correctness). Mitigated by keeping α ≤ λ.
- All covered by the λ ablation experiment plus a planned "stack-all" combined run (`λ=0.5, α=0.25`).

## Ablation Plan (λ=0 replaces R_correctness-only baseline; α=0 throughout the sweep)
| Condition | Equation | Purpose |
|---|---|---|
| λ=0 (`grpo_no_fairness.py`) | `R_total = -P_structural` | Does GRPO alone improve fairness? |
| λ=0.1 | `R_total = 0.1·R_fairness - P_structural` | Minimum fairness signal |
| λ=0.3 | `R_total = 0.3·R_fairness - P_structural` | Conservative fairness |
| λ=0.5 (**default**) | `R_total = 0.5·R_fairness - P_structural` | Main experiment (sweep variant) |
| λ=0.7 | `R_total = 0.7·R_fairness - P_structural` | Aggressive fairness |
| λ=1.0 | `R_total = R_fairness - P_structural` | Maximum fairness signal |
| **λ=0.5, α=0.25** (`fair_rlvr.yaml`) | `R_total = 0.5·R_fairness + 0.25·R_consistency - P_structural` | Headline run with counterfactual consistency on |
