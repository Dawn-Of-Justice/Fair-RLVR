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
- **Fix:** DAPO Clip-Higher (clip_ratio_high=0.28), entropy bonus in reward, Dynamic Sampling

### Risk 2: Over-Abstention / Lobotomy Effect (HIGH)
- **What:** λ too high → model learns to say "Unknown" to everything
- **Signs:** abstention rate >60%, MMLU drops significantly
- **Fix:** λ ablation; add abstention penalty if rate exceeds threshold; start λ=0.3 not 0.5
- **⚠️ Additional issue found:** The abstention metric itself was broken — `compute_abstention_rate()` was checking if the raw answer text (e.g., `"(c)"`) contained the word `"unknown"`, which never matched. Result was always 0% abstention regardless of actual behavior. **Fixed:** the function now uses a per-question `unknown_label` index field (set by `get_unknown_label()` in `data.py`) and does an index comparison. All early abstention metrics should be treated as invalid.

### Risk 3: Reward Hacking - Fairness Theater (MEDIUM)
- **What:** model learns to say "I must avoid bias" without real reasoning
- **Signs:** high BBQ accuracy but Faithfulness Score is low
- **Fix:** β penalty; Faithfulness Test (Experiment 4); strict verifier checks final answer only

### Risk 4: Small Model Bias Amplification (MEDIUM)
- **What:** FairReason showed 3B models can get MORE biased as reasoning improves
- **Signs:** bias score increases at intermediate training steps
- **Fix:** monitor bias score every 100 steps; if amplification occurs, lower lr or increase β

### Risk 5: Response Length Collapse (LOW-MEDIUM)
- **What:** VMR-RLVR showed structured task training collapses generation length
- **Signs:** <think> blocks shrink to 1-2 sentences
- **Fix:** add length reward component (min 50 tokens in <think>); mix 10% standard data

### Risk 6: T4 OOM (LOW)
- **What:** 16GB VRAM exceeded during training
- **Fix:** 4-bit quantization (unsloth), gradient checkpointing, reduce batch size, reduce max_new_tokens from 512→256

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
