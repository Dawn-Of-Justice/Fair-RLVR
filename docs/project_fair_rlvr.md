---
name: Fair-RLVR IEEE Paper Project
description: Accepted IEEE paper on training reasoning models to be fair via verifiable reward signals
type: project
---

# Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals

**Status:** Accepted by IEEE, not started yet (proposal stage as of 2026-03-18)
**Timeline:** 2.5 months (~10-11 weeks) from start

## Core Idea
Apply RLVR (Reinforcement Learning from Verifiable Rewards) — the same method behind DeepSeek-R1's emergent reasoning — to fairness/bias. Use BBQ (Bias Benchmark for QA, 58K labeled examples) as the verifiable ground truth for fairness, replacing subjective RLHF human feedback.

## Reward Function
```
R_total = R_correctness + λ·R_fairness - β·R_hacking_penalty
```
- R_correctness: format compliance + answering
- R_fairness: BBQ benchmark label match
- R_hacking_penalty: penalizes "fairness buzzwords" without real reasoning

## Technical Stack (planned)
- Model: Qwen2.5-3B-Instruct
- Training: GRPO (Group Relative Policy Optimization) + LoRA
- Hardware: Single T4 GPU (16GB), 4-bit quantization
- Dataset: BBQ (~1,000 training samples to start)
- Duration: ~6 weeks for training experiments

## Expected Deliverables
A) DatasetBench benchmark suite — pipeline comparing datasets across model sizes, token budgets, eval benchmarks
B) Dataset quality metrics — semantic redundancy, diversity/coverage, effective sample size, contamination checks
C) Data reduction + rewriting experiments — semantic dedup, LLM-based transformations
D) Publishable research paper — controlled results showing dataset properties predict training outcomes

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
