---
name: Key Papers Quick Reference
description: One-line summaries and arxiv IDs for all 12 literature review papers
type: reference
---

# Key Papers Quick Reference

## Must-Cite (Core Method)
| Paper | ID | One-line |
|---|---|---|
| DeepSeek-R1 | 2501.12948 | Pure RL causes emergent reasoning; RLVR foundation |
| DeepSeekMath / GRPO | 2402.03300 | GRPO algorithm: no critic, 50% less VRAM |
| DAPO | 2503.14476 | GRPO fixes: Clip-Higher + Dynamic Sampling, prevents entropy collapse |
| BBQ Benchmark | (dataset paper) | 58K fairness QA, 9 social dims, trinary choice, ground truth labels |

## Must-Cite (Fairness)
| Paper | ID | One-line |
|---|---|---|
| FairReason | 2507.23067 | GRPO for fairness in MLLMs; 1:4 ratio sweet spot; small models risk amplification |
| RealSafe-R1 | 2504.10081 | Safety alignment via deliberative RL; risk identified in <think> block |
| RLHF Sycophancy | 2602.01002 | RLHF amplifies agreement bias; worse in larger models |
| ASCL | 2602.13562 | Safety as tool-use; IFPO avoids over-refusal; adaptive not memorized |

## Must-Cite (Theory / Justification)
| Paper | ID | One-line |
|---|---|---|
| C2BMs | 2503.04363 | Causal bottleneck models; 64% demographic parity reduction |
| MCSQ Framework | (dataset paper) | Dataset quality: structure > size; ~10K sample peak |
| VMR-RLVR | 2511.02463 | RLVR extended to open-ended via multiple-choice reformulation |
| RLHF Algorithmic Bias | 2405.16455 | KL regularization → preference collapse; motivates moving away from RLHF |

## Must-Cite (Implementation Precedents)
| Paper | File | One-line |
|---|---|---|
| Med-RLVR | 2502.19655 | RLVR for medical MCQA on Qwen2.5-3B; documents 6 training phases; +8pts OOD over SFT |
| Reward Hacking / VCR | 2509.15557 (Tarek et al.) | Composite reward with P_structural + P_answer; ~85% reduction in hacking in Med-RLVR |

## Useful Quotes / Stats for Paper
- "R1-Zero: 15.6% → 71% on AIME 2024 via pure RL" (DeepSeek-R1)
- "GRPO reduces VRAM by ~50% vs PPO" (DeepSeekMath)
- "Models 3.4pp more accurate when answer aligns with stereotype" (BBQ)
- "1:4 bias:reasoning ratio → 10% stereotype reduction, 88% reasoning retained" (FairReason)
- "DAPO: 50 pts on AIME, 50% fewer steps than R1-Zero-32B" (DAPO)
- "C2BM: 64% reduction in age/gender demographic parity in medical systems" (C2BMs)
- "RLHF sycophancy scales negatively — worse in larger models" (Sycophancy)
- "Med-RLVR: RLVR not just for math/code — extends to medical MCQA; +8pts OOD vs SFT" (Med-RLVR)
- "Composite reward reduced reward hacking ~85% in medical RLVR setting" (Tarek et al.)
