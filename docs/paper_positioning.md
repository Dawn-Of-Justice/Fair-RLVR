---
name: Fair-RLVR Paper Positioning and Related Work Gap
description: How Fair-RLVR differs from existing work; talking points for Related Work section
type: project
---

# Paper Positioning

## The Core Claim
> Fair-RLVR is the first work to use a verifiable fairness benchmark (BBQ) as the primary RLVR reward signal for text reasoning models, and to analyze the causal faithfulness of the resulting reasoning chains.

## Comparison Table: Fair-RLVR vs. Existing Work

| Paper | Method | Domain | Fairness Ground Truth | Causal Faithfulness | Text-only |
|---|---|---|---|---|---|
| DeepSeek-R1 | RLVR | Math/Code | ✗ | ✗ | ✓ |
| Med-RLVR | RLVR | Medicine | ✗ | ✗ | ✓ |
| FairReason [2507.23067] | GRPO | Multimodal | BBQ (partial) | ✗ | ✗ |
| RealSafe-R1 [2504.10081] | SFT | Safety | Human-defined | ✗ | ✓ |
| ASCL [2602.13562] | IFPO | Safety | Human-defined | ✗ | ✓ |
| **Fair-RLVR (ours)** | GRPO+RLVR | Fairness | **BBQ (primary)** | **✓** | **✓** |

## Key Differentiators from FairReason (Closest Work)
1. **Text-only vs multimodal** — FairReason uses multimodal models; we focus on text LLMs
2. **BBQ as primary signal** — FairReason uses BBQ as *evaluation*; we use it as the *training verifier*
3. **Causal faithfulness analysis** — FairReason doesn't measure if CoT is real or post-hoc
4. **Reward hacking penalty** — FairReason doesn't have an explicit anti-theater mechanism
5. **DAPO stabilization** — FairReason uses vanilla GRPO; we incorporate Clip-Higher/Dynamic Sampling

## Key Differentiators from RLHF-based work (ASCL, RealSafe-R1)
- They use human-curated trajectories or human preference data → subjective, expensive
- Fair-RLVR uses automated verifiable labels → scalable, objective, no human bias in signal

## Narrative Arc for Related Work Section
1. RLVR works for objective tasks (DeepSeek-R1, DeepSeekMath)
2. RLVR extended to medical domain (Med-RLVR) → domain-specific reasoning emerges
3. RLVR extended to open-ended tasks (VMR-RLVR) → verifiable reformulation works
4. Fairness is treated as subjective (RLHF papers) → human labels are noisy and biased
5. BBQ proves fairness has ground truth → **gap: nobody used BBQ as RLVR signal**
6. FairReason hints at this but is multimodal and doesn't analyze faithfulness → **our work fills this gap**

## Claims to Defend in Experiments
1. Fair-RLVR reduces BBQ bias scores vs baseline (primary result)
2. Fair-RLVR does NOT degrade MMLU/GSM8K (no alignment tax)
3. Fair-RLVR reasoning is causally faithful (interventional sufficiency test)
4. Fair-RLVR generalizes to WinoBias/StereoSet (out-of-distribution fairness)
5. DAPO fixes (Clip-Higher) improve training stability over vanilla GRPO
