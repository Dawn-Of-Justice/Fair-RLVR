---
name: Fair-RLVR Paper Structure
description: Planned paper outline, section order, and key content per section
type: project
---

# Paper Outline: Fair-RLVR

**Venue:** IEEE (accepted)
**Target length:** 8-10 pages (IEEE double-column)

---

## Abstract (~150 words)
- Problem: RLHF is fragile — noisy human labels, vibe-based alignment
- Insight: Fairness has verifiable ground truth (BBQ benchmark)
- Method: GRPO with composite reward `R_total = λ·R_fairness + α·R_consistency - P_structural`
- Result: X% bias reduction, no alignment tax on MMLU/GSM8K
- Contribution: first text-only RLVR approach using fairness as verifiable reward

> ⚠️ Reward equation in abstract must be updated: old equation was `R_correctness + λ·R_fairness - β·R_hacking`. Current correct equation is `λ·R_fairness + α·R_consistency - P_structural` (R_consistency added per Ravulu et al. 2024 CDA, RLVR-adapted; off at α=0 in the lambda sweep, on at α=0.25 in the main `fair_rlvr.yaml` run).

One-liner pitch:
"While DeepSeek-R1 proved that reasoning emerges from correctness rewards in objective tasks, we demonstrate that fairness itself is a verifiable ground truth — and that models can spontaneously develop de-biasing logic within their chain-of-thought."

---

## 1. Introduction
- Hook: DeepSeek-R1's emergent reasoning via RL
- Problem: RLHF's three failures (vibe-based, lobotomy, fake thinking)
- Key insight: BBQ makes fairness verifiable → RLVR applies
- Contributions (bullet list):
  1. Fair-RLVR framework with composite reward
  2. Empirical evidence of emergent de-biasing reasoning
  3. Causal faithfulness analysis of CoT
  4. No alignment tax: fairness ↑, reasoning maintained

---

## 2. Background & Related Work
- 2.1 RLVR and GRPO (DeepSeek-R1, DeepSeekMath, DAPO)
- 2.2 Fairness benchmarks (BBQ)
- 2.3 RLHF limitations (Sycophancy paper, Xiao et al. 2024)
- 2.4 Safety alignment via RL (RealSafe-R1, ASCL)
- 2.5 Fairness + RL (FairReason — closest, key differentiator)

---

## 3. Method
- 3.1 Problem Formulation — define fairness as verifiable task
- 3.2 BBQ as Fairness Verifier — ambiguous vs disambiguated contexts
- 3.3 Reward Function — full R_total formula with component explanations
- 3.4 Training Setup — GRPO + LoRA + DAPO improvements
- 3.5 Thought Template — system prompt for <think>/<answer> format

---

## 4. Experiments
- 4.1 Setup (model, hardware, datasets, baselines)
- 4.2 Main Results (Experiment 1 table)
- 4.3 Lambda Ablation (Experiment 2)
- 4.4 Training Dynamics — the 6 phases analog (Experiment 3)
- 4.5 Causal Faithfulness (Experiment 4)
- 4.6 OOD Generalization (Experiment 5)
- 4.7 Adversarial Stress Test

---

## 5. Analysis
- 5.1 Emergent De-biasing Logic — CoT examples showing "catching" bias
- 5.2 Why Fair Logic Beats Fair Results — theoretical argument
- 5.3 Small Model Warning — address FairReason's finding (Exp 6)
- 5.4 Mathematical Justification — mutual information, PAC-learning bound, causal bottleneck

---

## 6. Limitations
- BBQ is Western-centric
- Single model size (3B) — no scaling study
- T4 compute limits longer training runs
- Faithfulness metric is approximate (not mechanistic)
- CoT may still be partially post-hoc

---

## 7. Conclusion
- Summary of contributions
- Fair-RLVR as a template: fairness is not a tax on intelligence
- Future: larger models, multilingual BBQ (BharatBBQ), automated causal graph discovery (C2BM direction)

---

## Appendix (if needed)
- Full prompt/thought template used in training
- Extended training curves (all lambda values)
- Additional CoT examples (good and failure cases)
- Dataset split details
