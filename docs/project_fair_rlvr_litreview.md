---
name: Fair-RLVR Literature Review
description: 12 papers forming the literature review for the Fair-RLVR IEEE paper
type: project
---

# Fair-RLVR Literature Review

## Foundational / Methodology Papers

1. **DeepSeek-R1** [2501.12948] — Pure RL training causes emergent reasoning (R1-Zero: 15.6%→71% AIME). Multi-stage with cold-start SFT fixes readability. *Basis for RLVR approach.*

2. **DeepSeekMath / GRPO** [2402.03300] — Introduces GRPO (Group Relative Policy Optimization). Eliminates critic model, ~50% memory reduction vs PPO. Risk: entropy collapse. *Core training algorithm.*

3. **RLVR Framework** — Binary verifiable rewards, no human preference models. Bias-free ground truth signal. Traditionally limited to objective domains. *Core training paradigm.*

4. **DAPO** [2503.14476] — Addresses GRPO instability: Clip-Higher + Dynamic Sampling. Maintains entropy during training. *Practical GRPO improvements to incorporate.*

## Fairness / Bias Papers

5. **BBQ Benchmark** — 58K trinary-choice QA across 9 social dimensions. Ambiguous (Unknown=correct) vs disambiguated contexts. Models are 3.4pp more accurate when answer aligns with stereotype. *Primary fairness verifier/dataset.*

6. **FairReason** [2507.23067] — RL (GRPO) > SFT > KD for bias mitigation in MLLMs. Sweet spot: 1:4 bias-to-reasoning data ratio → 10% stereotype reduction, 88% reasoning retained. Key finding: small models can get *more* biased with improved reasoning. *Direct precursor — closest existing work.*

7. **RealSafe-R1** [2504.10081] — Safety-aligns R1 via deliberative alignment: 15K safety-aware reasoning trajectories. Model identifies risks *inside* thought block before refusing. *Parallel work in safety domain.*

8. **ASCL** [2602.13562] — Safety as multi-turn tool-use; IFPO rebalances over-refusal. Decouples rules from reasoning path, avoids lobotomy effect. *Relevant to avoiding refusal bias.*

## Theoretical / Mechanistic Papers

9. **C2BMs** [2503.04363] — Causally Reliable Concept Bottleneck Models. 64% reduction in demographic parity differences in medical systems via causal intervention. *Mathematical grounding for causal bottleneck argument.*

10. **RLHF Sycophancy** [2602.01002] — RLHF amplifies agreement bias; scales negatively (worse in larger models). KL divergence "agreement penalties" as mitigation. *Why RLHF fails; motivation for RLVR approach.*

11. **MCSQ Framework** — Dataset quality: Volume, Scope, Granularity, Variety, Distortion, Mismatch. Structure > size. Performance peaks ~10K samples. Distorted (model-disagreed) samples can help. *Dataset selection guidance.*

12. **VMR-RLVR** [2511.02463] — Extends RLVR to open-ended tasks via multiple-choice reformulation. +3.29pts over reward-model RL. Training collapse risk (response length). *Shows RLVR can work beyond math/code.*

## Key Gaps This Paper Fills
- FairReason is closest but: multimodal, no explicit causal bottleneck analysis, no faithfulness metric
- No paper has used BBQ as the *primary* verifiable reward signal in GRPO training
- No paper has measured whether fairness reasoning is *causally faithful* vs post-hoc
- No paper has proven "no alignment tax" formally — Fair-RLVR claims fairness improves reasoning
