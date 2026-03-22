---
name: Fair-RLVR Reward Function Design
description: Detailed breakdown of the composite reward function with structural and semantic anti-hacking penalties
type: project
---

# Reward Function: R_total

```
R_total = R_correctness + λ·R_fairness - P_structural - P_leak
```

## Components

### R_correctness
- Did the model use `<think>` and `<answer>` tags correctly?
- Did it produce a valid answer (not blank, not gibberish)?
- Binary: +1 (correct format + valid answer), -1 for missing/malformed tags
- Mirrors Med-RLVR's format reward: -1 for missing tags

### R_fairness
- Does the final answer match the BBQ ground truth label?
- BBQ labels: "Unknown" (ambiguous context) or specific answer (disambiguated context)
- Binary: +1 for match, 0 for mismatch
- λ controls the weight — needs tuning (start: λ=0.5)
- FairReason found 1:4 bias:reasoning ratio is optimal sweet spot

### P_structural (structural penalty)
Rule-based, checks three violations (each = 0.3 penalty):
1. Answer leaking into `<think>` block
2. Empty or trivially short reasoning (< 20 tokens)
3. Reasoning content outside designated tags

Based on: Tarek et al. "Reward Hacking Mitigation using Verifiable Composite Rewards" (2025)
- Reduced hacking by ~85% in Med-RLVR setting

### P_leak (semantic leak penalty)
- Uses Sentence-BERT embeddings to compute cosine similarity between `<think>` and `<answer>`
- If similarity > threshold τ (default 0.85), penalty = sim - τ
- Catches model restating answer as "reasoning" without real logic
- Fully automated, no keyword lists needed

Based on: Tarek et al. (2025) extended variant

## Why NOT keyword-based R_hacking
- Manually picking "fairness buzzwords" is subjective — same flaw as RLHF
- Penalizes legitimate reasoning that uses words like "stereotype" or "bias"
- Contradicts our paper's argument for automated, reproducible signals

## Tuning Risks
- λ too high → "lobotomy effect": model says "Unknown" to everything (over-abstention)
- λ too low → fairness signal drowned out by correctness reward
- τ too low → penalizes legitimate short reasoning that naturally overlaps with answer
- τ too high → fails to catch semantic leaking

## Planned Ablations
1. R_correctness only (baseline)
2. R_correctness + R_fairness (no penalties)
3. Full R_total (with P_structural + P_leak)
4. λ sweep: {0.1, 0.3, 0.5, 0.7, 1.0}
