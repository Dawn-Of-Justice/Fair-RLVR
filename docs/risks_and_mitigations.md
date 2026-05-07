---
name: Fair-RLVR Risks and Mitigations
description: Known risks, failure modes, and how to handle them
type: project
---

# Risks and Mitigations

## Technical Risks

### Risk 1: Entropy Collapse (HIGH)
- **What:** GRPO narrows policy → all outputs identical → no learning
- **Signs:** reward plateau early, all G samples give same answer
- **Fix (implemented):** DAPO Clip-Higher (`epsilon_high=0.28`, `epsilon_low=0.20`); GRPOTrainer's dynamic sampling filters all-same-reward groups automatically

### Risk 2: Over-Abstention / Lobotomy Effect (HIGH)
- **What:** λ too high → model learns to say "Unknown" to everything (safe-looking but wrong on disambiguated questions)
- **Signs:** abstention rate >60% on disambiguated contexts, MMLU drops
- **Fix (implemented):** λ ablation sweep; disambiguated accuracy tracked separately to catch this early
- **⚠️ Abstention metric was broken and is now fixed:** `compute_abstention_rate()` now uses per-question `unknown_label` index (set by `get_unknown_label()` in `data.py`) instead of a string heuristic. All pre-fix abstention metrics are invalid.

### Risk 3: Reward Hacking - Fairness Theater (MEDIUM)
- **What:** model learns to insert bias-aware language without genuine reasoning
- **Signs:** high BBQ accuracy but low interventional sensitivity score (Experiment 4)
- **Fix (implemented):** `P_structural` penalizes trivially short reasoning; Experiment 4 (sentence-permutation test) checks whether behavior is reasoning-dependent

### Risk 4: Small Model Bias Amplification (MEDIUM)
- **What:** FairReason showed 3B models can get MORE biased as reasoning improves
- **Signs:** official BBQ bias score increases at intermediate training steps
- **Fix:** monitor `bias_score_bbq` per checkpoint; if amplification observed, reduce learning rate

### Risk 5: Response Length Collapse (LOW-MEDIUM)
- **What:** VMR-RLVR showed structured task training collapses generation length
- **Signs:** `<think>` blocks shrink to 1-2 sentences; P_structural short-reasoning penalty fires frequently
- **Fix:** `P_structural` already penalizes `<20` token reasoning chains; monitor think-block length in CoT samples

### Risk 6: H100 OOM (LOW)
- **What:** 80GB VRAM exceeded (unlikely at current config, possible at larger batch)
- **Fix (implemented):** 4-bit quantization (bitsandbytes nf4), gradient checkpointing; fall back to `--batch-size 4 --grad-accum 4` if needed

## Research/Paper Risks

### Risk 7: FairReason Scoops Contribution (MEDIUM)
- **What:** FairReason [2507.23067] overlaps significantly
- **Mitigation:** emphasize text-only, causal faithfulness, BBQ as primary training signal (not just eval), DAPO integration — these are clear gaps

### Risk 8: Reviewers Question Faithfulness Metric (MEDIUM)
- **What:** interventional sufficiency is not mechanistic proof
- **Mitigation:** be transparent about limitation; cite C2BM paper for theoretical grounding; future work: Logit Lens / probing

### Risk 9: BBQ Train/Eval Contamination → **RESOLVED**
- **What:** training and evaluating on BBQ could inflate scores
- **Fix implemented:** strict seed-based 90/10 split via `create_splits(train_ratio=0.9, seed=42)`. The seed is threaded through all scripts — `train.py`, `evaluate.py`, `baseline_model.py`, `sft.py`, `grpo_no_fairness.py` all accept `--seed` and default to 42. Mismatched seeds were a source of data leakage — now caught and fixed.
- OOD evaluation on WinoBias + StereoSet ensures the BBQ split isn't the only evidence of generalization.

### Risk 10: No Significant Result (LOW but serious)
- **What:** Fair-RLVR doesn't beat baselines on BBQ
- **Mitigation:** even null results are publishable if methodology is sound; frame as "fairness is harder to emerge than math reasoning" — still a contribution

### Risk 11: "Unknown" Option Position Varies Per Question (DISCOVERED)
- **What:** The "Unknown/Can't be determined" answer option is NOT always option (c). It can be (a), (b), or (c) depending on the question. Early code hardcoded `answer_to_index(answer) == 2` (i.e., always option c) for abstention detection in callbacks.py.
- **Impact:** All abstention measurements in the callback during training were wrong (either always 0% or wrong rate).
- **Fix implemented:** `get_unknown_label(example)` in `data.py` scans `ans0/ans1/ans2` text for "unknown", "cannot be determined" etc. and returns the correct index. The `unknown_label` field is attached to every example. All downstream code (`evaluate.py`, `callbacks.py`, all baselines) now use this index for abstention detection.

## Reviewer Questions to Pre-empt
1. "How is this different from FairReason?" → text-only, BBQ as training verifier, faithfulness
2. "Why 3B? Results won't scale." → feasibility study; cite Med-RLVR using 3B; future work
3. "BBQ is Western-centric — how do results generalize?" → acknowledge as limitation; future work: BharatBBQ
4. "Is the CoT real or fake?" → Experiment 4 (faithfulness test) directly addresses this
5. "What's the lambda sensitivity?" → Experiment 2 (lambda ablation) directly addresses this
