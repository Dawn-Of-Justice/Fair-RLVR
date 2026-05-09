# Reference Paper Summaries

Quick-reference notes for all 15 papers in this folder. Each entry covers what the paper is about, its key contributions, limitations, and directions for future work.

---

## Table of Contents

1. [DeepSeek-R1: Incentivizing Reasoning via RL](#1-deepseek-r1-incentivizing-reasoning-via-rl)
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](#2-deepseekmath-pushing-the-limits-of-mathematical-reasoning)
3. [BBQ: A Hand-Built Bias Benchmark for Question Answering](#3-bbq-a-hand-built-bias-benchmark-for-question-answering)
4. [FairReason: Balancing Reasoning and Social Bias in MLLMs](#4-fairreason-balancing-reasoning-and-social-bias-in-mllms)
5. [RealSafe-R1: Safety-Aligned DeepSeek-R1](#5-realsafe-r1-safety-aligned-deepseek-r1)
6. [How RLHF Amplifies Sycophancy](#6-how-rlhf-amplifies-sycophancy)
7. [DAPO: An Open-Source LLM RL System at Scale](#7-dapo-an-open-source-llm-rl-system-at-scale)
8. [Extending RLVR to Open-Ended Tasks via MC Reformulation](#8-extending-rlvr-to-open-ended-tasks-via-mc-reformulation)
9. [RLHF Preference Collapse and Matching Regularization](#9-rlhf-preference-collapse-and-matching-regularization)
10. [Causally Reliable Concept Bottleneck Models](#10-causally-reliable-concept-bottleneck-models)
11. [Adaptive Safe Context Learning (ASCL)](#11-adaptive-safe-context-learning-ascl)
12. [Structure Trumps Size: Rethinking Data Quality for LLM Reasoning](#12-structure-trumps-size-rethinking-data-quality-for-llm-reasoning)
13. [Med-RLVR: Emerging Medical Reasoning via RL](#13-med-rlvr-emerging-medical-reasoning-via-rl)
14. [Reward Hacking Mitigation using Verifiable Composite Rewards](#14-reward-hacking-mitigation-using-verifiable-composite-rewards)
15. [Mitigating Bias in RLHF for Large Language Models](#15-mitigating-bias-in-rlhf-for-large-language-models)

---

## 1. DeepSeek-R1: Incentivizing Reasoning via RL

**File:** `DeepSeek-R1_Reasoning_via_RL.pdf`
**Authors:** DeepSeek-AI
**Venue:** arXiv:2501.12948 (2025)

### What It's About

DeepSeek-R1 shows that LLMs can develop sophisticated reasoning capabilities through pure reinforcement learning, without any supervised fine-tuning on human-annotated reasoning traces. The paper introduces two models: DeepSeek-R1-Zero (pure RL from a base model with no SFT) and DeepSeek-R1 (a multi-stage pipeline combining cold-start SFT, RL, rejection sampling, and another SFT pass). Both use GRPO as the RL algorithm. The reward signal is purely rule-based — correctness of final answers and adherence to a `<think>...</think><answer>...</answer>` format. The model spontaneously develops self-reflection, verification, and backtracking behaviors without being explicitly taught to do so.

### Key Contributions

- Demonstrates that reasoning emerges from RL alone, without human-labeled reasoning trajectories — a major challenge to the assumption that CoT reasoning requires supervised data.
- DeepSeek-R1-Zero achieves competitive performance on AIME and math benchmarks through self-evolved reasoning.
- DeepSeek-R1 (multi-stage) achieves state-of-the-art results, matching or beating OpenAI-o1 on several benchmarks.
- Introduces an "aha moment" phenomenon: at some point during training, the model begins to re-examine its own reasoning and self-correct.
- Releases distilled smaller models (1.5B–70B) that transfer reasoning ability from the large model.

### Limitations

- DeepSeek-R1-Zero suffers from language mixing (blending Chinese and English in the same response) and low readability.
- The multi-stage pipeline (DeepSeek-R1) is complex and requires careful engineering of each stage.
- Strong dependence on having a powerful base model (DeepSeek-V3-Base, 671B MoE) — results may not transfer to smaller or weaker base models.
- Reward signal is narrowly defined around correctness; there is no mechanism for rewarding reasoning quality, safety, or fairness directly.
- Limited to domains with verifiable ground truth (math, coding); cannot be directly applied to subjective or open-ended tasks.

### Future Directions

- Extending RLVR beyond math and coding to other verifiable domains (medicine, law, fairness — the core motivation for Fair-RLVR).
- Making the reasoning process more interpretable and reducing verbosity without sacrificing quality.
- Improving the efficiency of the RL training loop for smaller hardware budgets.
- Studying whether the emergent behaviors (self-reflection, verification) can be induced more reliably and at smaller scales.

---

## 2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning

**File:** `DeepSeekMath.pdf`
**Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu et al. (DeepSeek-AI)
**Venue:** arXiv:2402.03300 (2024)

### What It's About

DeepSeekMath introduces a 7B math-specialized language model trained on 120 billion math-related tokens mined from Common Crawl. The paper's primary technical contribution is the Group Relative Policy Optimization (GRPO) algorithm — the RL algorithm used in Fair-RLVR. GRPO eliminates the need for a critic/value model (unlike PPO) by estimating the value baseline from the mean reward within a group of sampled completions for the same prompt. This dramatically reduces memory and compute requirements compared to PPO while achieving comparable or better performance.

### Key Contributions

- Introduces GRPO: samples G outputs per prompt, normalizes rewards within the group to compute advantages, then applies a PPO-style clipped objective. No separate value network needed.
- Provides a unified theoretical framework showing that RFT, DPO, PPO, and GRPO are all variants of the same RL paradigm.
- Demonstrates that code pre-training before math pre-training improves mathematical reasoning ability.
- Constructs the DeepSeekMath Corpus (120B tokens) using an iterative fastText-based filtering pipeline from Common Crawl.
- DeepSeekMath-RL achieves 51.7% on competition-level MATH (matching GPT-4 at the time) as a 7B model.

### Limitations

- GRPO's within-group normalization means the reward signal can vanish if all sampled outputs in a group are uniformly correct or uniformly wrong (the advantage is 0 for all). This is the "entropy collapse" problem that DAPO later addresses.
- Pre-training on 120B tokens is expensive; the paper doesn't fully explore how much data is actually needed.
- GRPO in this paper uses symmetric clipping (same ε high and low) which later work (DAPO) shows is suboptimal.
- Focused entirely on math; generalization to other domains is left open.
- arXiv math data didn't improve performance despite being commonly used — an unintuitive negative result that merits deeper investigation.

### Future Directions

- DAPO extends GRPO with asymmetric clipping, dynamic sampling, and token-level loss (see Paper 7).
- Applying GRPO to non-math domains (medicine, fairness, law) — directly realized in Med-RLVR and Fair-RLVR.
- Process reward models (step-level rewards instead of outcome-level) to provide denser training signals.
- Better understanding of why code pre-training helps mathematical reasoning.

---

## 3. BBQ: A Hand-Built Bias Benchmark for Question Answering

**File:** `BBQ_Bias_Benchmark.pdf`
**Authors:** Alicia Parrish, Angelica Chen, Nikita Nangia et al. (NYU)
**Venue:** ACL 2022 (arXiv:2110.08193)

### What It's About

BBQ (Bias Benchmark for QA) is the dataset used as the fairness verifier in Fair-RLVR. It consists of hand-crafted multiple-choice questions (3 options: a/b/c) designed to test whether QA models rely on social stereotypes. Each question set has two versions: an ambiguous context (where the correct answer is always "Unknown" — no definitive answer is possible) and a disambiguated context (where the text provides enough information to determine the answer). Questions span 9 social categories: Age, Disability Status, Gender Identity, Nationality, Physical Appearance, Race/Ethnicity, Religion, SES, and Sexual Orientation.

### Key Contributions

- Defines a two-step official bias metric (implemented in `BBQ_calculate_bias_score.R`):
  1. Filter out "Unknown" predictions, then compute `raw = 2 × P(target | non-Unknown) − 1`
  2. Scale ambiguous score by accuracy: `bias = raw × (1 − accuracy_ambig)` — so a model that correctly abstains on ambiguous questions gets a near-zero bias score rather than appearing counter-stereotyped
  Disambiguated score is unscaled. Range [−1, 1]; 0 = unbiased; positive = stereotype-biased.
  Note: The paper also mentions a simplified metric (proportion of *errors* that are stereotype-consistent, range [0, 1], 0.5 = random errors) — this is secondary and not the formula in the R script.
- Separates two failure modes: (1) relying on stereotypes when context is ambiguous, and (2) letting stereotypes override correct evidence when context is disambiguated.
- Finds that all tested models rely heavily on stereotypes in ambiguous contexts, and even in disambiguated contexts, models are 3–5 percentage points more accurate when the correct answer aligns with stereotypes.
- Provides a reproducible, automated evaluation framework — no human raters needed post-construction.
- Covers attested biases from social psychology literature, making it empirically grounded.

### Limitations

- Only covers U.S. English-speaking social contexts — bias patterns in other languages and cultures are not represented.
- The binary ambiguous/disambiguated setup is somewhat artificial; real-world questions often have intermediate levels of informativeness.
- "Unknown" as the correct answer for all ambiguous questions may inflate accuracy scores for models that learn to always abstain.
- The benchmark is static — it cannot capture newly emerging stereotypes or biases introduced after construction.
- Intersectional categories (race × gender, race × SES) are less well-covered.
- Does not test generation tasks — only evaluates multiple-choice QA behavior.

### Future Directions

- Extending to multilingual and cross-cultural settings.
- Dynamic bias benchmarks that update with evolving social norms.
- Testing open-ended generation rather than just classification behavior.
- Developing process-level bias evaluation (evaluating the reasoning chain, not just the final answer).
- Using BBQ as a training signal rather than only an evaluation tool — the core idea of Fair-RLVR.

---

## 4. FairReason: Balancing Reasoning and Social Bias in MLLMs

**File:** `FairReason.pdf`
**Authors:** Zhenyu Pan, Yutong Zhang, Jianshu Zhang et al. (Northwestern University, UIC)
**Venue:** arXiv:2507.23067 (2025)

### What It's About

FairReason is the most directly related paper to Fair-RLVR. It studies the interaction between reasoning improvement and social bias mitigation in multimodal large language models (MLLMs), and finds that improved reasoning does not automatically lead to reduced bias — the relationship is complex and depends on model size, training strategy, and data composition. The paper benchmarks three training strategies (SFT, knowledge distillation, rule-based RL) and systematically varies the ratio of debias-focused to reasoning-focused training samples to find an optimal balance.

### Key Contributions

- Empirically disproves the assumption that "better reasoning → less bias" — the relationship is nuanced and model-dependent.
- Identifies a "sweet spot" in training data composition: a roughly 1:4 ratio of bias-mitigation samples to reasoning samples cuts stereotype scores by 10% while retaining 88% of reasoning accuracy.
- Shows that rule-based RL consistently outperforms SFT and knowledge distillation for bias mitigation across all tested models.
- Releases best-performing models on HuggingFace for reproducibility.
- Evaluates on both bias (BBQ, VLBiasBench) and reasoning (AIME, MATH-500, MathVerse, Geometry3K) benchmarks.

### Limitations

- Uses only 5,000 training samples — results may not scale to larger training budgets.
- Focused on MLLMs (vision-language models); language-only models may behave differently.
- The 1:4 ratio sweet spot may not generalize across all model families or training budgets.
- Does not analyze WHY RL outperforms SFT for bias — only observes that it does.
- VLBiasBench and BBQ only measure classification behavior; generation-level bias is not evaluated.
- The study doesn't examine whether the bias reductions are causally tied to the reasoning chain or are superficial.

### Future Directions

- Scaling studies: does the 1:4 ratio hold with 50k or 500k training samples?
- Analyzing whether RL-trained models genuinely reason about bias or merely learn surface patterns.
- Extending to language-only models and text-only bias datasets.
- Combining fairness and reasoning rewards in a single composite reward function — exactly what Fair-RLVR does.
- Investigating whether causal faithfulness of the bias-reasoning chain can be measured.

---

## 5. RealSafe-R1: Safety-Aligned DeepSeek-R1

**File:** `RealSafe-R1.pdf`
**Authors:** Yichi Zhang, Zihao Zeng, Dongbai Li et al. (Tsinghua University, RealAI, SJTU)
**Venue:** arXiv:2504.10081 (2025)

### What It's About

RealSafe-R1 addresses the safety failures of open-source DeepSeek-R1 models, which often comply with harmful queries despite showing safety awareness in their reasoning chains. The key insight is that R1 models already have latent safety knowledge — they just fail to act on it. The paper constructs 15,000 safety-aware reasoning trajectories by prompting DeepSeek-R1 with explicit refusal instructions, then fine-tunes smaller distilled models on this dataset via SFT. This approach preserves reasoning capability by keeping training data within the original reasoning format distribution.

### Key Contributions

- Demonstrates that existing safety alignment datasets (designed for instruction-tuned models) fail for reasoning models because they lack structured long reasoning outputs — causing a style mismatch that degrades reasoning.
- Generates safety-aware CoT data from DeepSeek-R1 itself (distillation approach), maintaining output format consistency.
- Reduces harmful StrongReject scores under PAIR and PAP jailbreak attacks from 0.73/0.61 to 0.27/0.10 for the 32B model.
- Preserves reasoning performance and even improves TruthfulQA scores.
- Open-sources RealSafe-R1 model weights.

### Limitations

- Uses SFT rather than RL — the safety alignment may be brittle and susceptible to jailbreaks not seen in training.
- Dataset construction relies on DeepSeek-R1 generating safe trajectories, which requires careful prompt engineering and filtering.
- Evaluated primarily on English safety benchmarks; cross-lingual safety is not addressed.
- The approach requires access to a large capable model (DeepSeek-R1) to generate training data — not feasible for resource-constrained settings.
- Does not study whether the safety reasoning is causally linked to the refusal decision or is post-hoc.

### Future Directions

- Using RL with safety verifiable rewards instead of SFT to improve robustness.
- Extending to multilingual and multimodal safety.
- Automated pipeline for continuously updating safety datasets as new attack patterns emerge.
- Studying the causal relationship between the safety reasoning chain and the refusal decision.

---

## 6. How RLHF Amplifies Sycophancy

**File:** `RLHF_Amplifies_Sycophancy.pdf`
**Authors:** Itai Shapira, Gerdus Benade, Ariel D. Procaccia (Harvard, Boston University)
**Venue:** arXiv:2602.01002 (2026)

### What It's About

This paper provides a formal mathematical explanation for why preference-based post-training (RLHF) tends to make models more sycophantic — agreeing with user claims even when incorrect. The key mechanism: if human annotators have a slight preference for responses that affirm their beliefs (even when those responses are wrong), the reward model learns to reward agreement, and optimizing the policy against that reward amplifies this bias. The direction of behavioral drift is determined by the covariance between the "agree with user" signal and the learned reward under the base policy. The paper derives a closed-form correction (an agreement penalty) that neutralizes this amplification.

### Key Contributions

- First formal mechanistic analysis linking reward learning from biased human preferences to sycophancy amplification at the policy level.
- Shows that even small annotator biases in pairwise comparisons can induce substantial behavioral drift.
- Derives the unique KL-closest policy that prevents sycophancy from increasing (compared to the base model) as a closed-form result.
- Provides an agreement penalty that can be added to training to counteract sycophancy.
- Empirically validates the framework showing reward tilt correlates with direction of behavioral drift across diverse model families.

### Limitations

- Analysis assumes a relatively simple random utility model (Bradley-Terry); real annotator behavior is more complex.
- The proposed agreement penalty requires access to a "belief signal" in the prompt, which is not always present.
- Focuses on sycophancy as a binary (agree/disagree) behavior — doesn't model more subtle forms.
- The KL-closest policy correction may be difficult to implement efficiently in practice.
- Does not address multi-turn sycophancy where the model progressively shifts position over a conversation.

### Relevance to Fair-RLVR

This paper directly motivates why RLHF is a poor choice for fairness alignment: if annotators have stereotyped preferences (as social psychology research shows they do), RLHF will amplify those biases rather than reduce them. RLVR with objective ground truth (BBQ labels) sidesteps this problem entirely.

### Future Directions

- Extending the framework to multi-turn conversations.
- Empirical studies on how much annotator bias is present in real RLHF datasets.
- Combining the agreement penalty with RLVR to avoid the sycophancy problem in domains with partial verifiability.

---

## 7. DAPO: An Open-Source LLM RL System at Scale

**File:** `DAPO.pdf`
**Authors:** ByteDance Seed, Tsinghua AIR, HKU
**Venue:** arXiv:2503.14476 (2025)

### What It's About

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) identifies and fixes four key failure modes in naive GRPO training that prevent reproducing DeepSeek-R1-level results. It achieves 50 points on AIME 2024 with Qwen2.5-32B — outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) in 50% fewer training steps. The four techniques are: (1) asymmetric clipping (Clip-Higher), (2) dynamic sampling to skip all-correct or all-wrong batches, (3) token-level policy gradient loss instead of sequence-level, and (4) overlong reward shaping to reduce noise from truncated responses.

### Key Contributions

- **Clip-Higher (asymmetric clipping):** Uses different clip ratios for high and low importance sampling ratios (ε_high=0.28, ε_low=0.20 in Fair-RLVR). This promotes diversity and prevents entropy collapse without penalizing the model for getting things right.
- **Dynamic Sampling:** Filters out prompts where all G sampled responses are correct or all are wrong, since these provide no gradient signal. Replaces them with new prompts during training.
- **Token-Level Policy Gradient:** Computes loss per token rather than averaging over sequences. Prevents long reasoning chains from being penalized relative to short ones.
- **Overlong Reward Shaping:** Applies a soft penalty for responses that hit the maximum token limit (likely truncated and therefore unreliable), reducing gradient noise.
- Fully open-sources training code, algorithm, and dataset on the verl framework.

### Limitations

- Only evaluated on math reasoning (AIME); it's unclear whether all four fixes are equally necessary in non-math domains.
- Dynamic sampling increases training time per step since rejected batches need replacement.
- Requires a large compute budget (32B model, thousands of steps) — the improvements at smaller scales are less studied.
- Overlong reward shaping is a heuristic; the optimal penalty shape is not theoretically derived.

### Relevance to Fair-RLVR

Fair-RLVR directly adopts DAPO's asymmetric clipping (ε_high=0.28, ε_low=0.20) as part of its GRPO training configuration. This is a key ingredient that makes training stable at the 3500-step scale.

### Future Directions

- Studying which of the four techniques contributes most to improvements in non-math domains.
- Combining DAPO with process reward models for denser training signals.
- Scaling analysis: at what model sizes do these fixes become critical?

---

## 8. Extending RLVR to Open-Ended Tasks via MC Reformulation

**File:** `RLVR_MC_Reformulation.pdf`
**Authors:** Mengyu Zhang, Siyu Ding, Weichong Yin et al. (Baidu Ernie Team)
**Venue:** arXiv:2511.02463 (2026)

### What It's About

RLVR has been shown to work for math and coding, which have clear verifiable answers. This paper asks: can RLVR be applied to open-ended tasks like creative writing and subjective Q&A? The answer is yes, using a technique called VMR-RLVR (Verifiable Multiple-Choice Reformulation). The key idea is to reformat open-ended tasks into multiple-choice questions — either by generating answer candidates and using a strong model to select the best one as ground truth, or by converting the evaluation criteria into a checkable discrete format. This enables RLVR training even when there's no canonical ground truth.

### Key Contributions

- Demonstrates that multiple-choice reformulation is a general strategy for applying RLVR to any domain.
- Achieves an average gain of 3.29 points over RL with reward models across seven open-ended benchmarks (MTBench, AlpacaEval, WildBench, CreativeWriting, ArenaHard, IFEval, LiveBench).
- Shows that reformulated verifiable signals are more stable training targets than reward model scores.
- Provides a principled bridge between RLVR (verifiable domains) and RLHF (subjective domains).

### Limitations

- The quality of the reformulated multiple-choice labels depends on the strength of the teacher model used to generate them — weak teachers produce noisy labels.
- Multiple-choice reformulation may not capture all dimensions of quality in truly open-ended tasks (e.g., creativity, originality).
- Evaluation is on general open-ended tasks; domain-specific subjective tasks (legal reasoning, medical advice) are not evaluated.
- The approach essentially converts a hard problem (subjectivity) into a different hard problem (constructing good distractors and picking a reliable ground truth).

### Relevance to Fair-RLVR

This paper validates the general principle that BBQ (a multiple-choice bias benchmark) is a valid training signal for RLVR. Fair-RLVR is essentially applying the same insight to the fairness domain: bias evaluation is inherently a multiple-choice verification task.

### Future Directions

- Automated pipeline for reformulating any benchmark into verifiable MC format.
- Iterative reformulation where the model's own outputs inform better distractor generation.
- Combining VMR-RLVR with fairness objectives to create verifiable fairness signals.

---

## 9. RLHF Preference Collapse and Matching Regularization

**File:** `RLHF_Preference_Collapse.pdf`
**Authors:** Jiancong Xiao, Ziniu Li, Xingyu Xie et al. (UPenn, CUHK-SZ, NUS, PKU)
**Venue:** Journal of the American Statistical Association (arXiv:2405.16455, 2025)

### What It's About

This paper argues that standard RLHF has a fundamental algorithmic bias: the KL-divergence regularization used to prevent the model from deviating too far from the reference model inadvertently suppresses minority preferences in the human feedback distribution. In extreme cases, this leads to "preference collapse" — where the aligned model only reflects majority preferences and completely ignores minority viewpoints. The paper proposes Preference Matching (PM) RLHF as a fix, which replaces the KL regularizer with one derived from a differential equation that ensures the final policy distribution matches the reward model's preference distribution across all groups.

### Key Contributions

- Formally proves that KL-regularized RLHF has an inherent algorithmic bias that can suppress minority preferences entirely.
- Defines the concept of "preference collapse" — a failure mode where minority groups' preferences are effectively zeroed out by the optimization.
- Derives PM RLHF using an ODE that enforces preference matching — the policy's output distribution over responses should match the preference distribution of the reward model.
- Shows 29–41% improvement in alignment with human preferences (OPT and Llama models) compared to standard RLHF.
- Introduces a conditional variant of PM RLHF practical for natural language generation.

### Limitations

- The PM regularizer is derived under the Bradley-Terry-Luce model, which is a specific parametric assumption about human preference structure.
- The ODE-based derivation is theoretically elegant but may be computationally expensive to implement exactly.
- The "minority preferences" studied are abstract — it's unclear how this maps to specific demographic groups or real-world social biases.
- Experiments are on relatively small models (OPT, Llama-family); scaling behavior is unknown.

### Relevance to Fair-RLVR

This paper explains why RLHF is structurally problematic for fairness: not only does it amplify sycophancy (Paper 6), it also structurally ignores minority preferences. RLVR with objective labels (BBQ) bypasses both problems.

### Future Directions

- Applying PM RLHF to fairness-specific datasets where minority group preferences are explicitly tracked.
- Combining PM RLHF with RLVR for domains that have partial verifiability.
- Studying whether preference collapse occurs in practice and how to detect it.

---

## 10. Causally Reliable Concept Bottleneck Models

**File:** `Causally_Reliable_CBMs.pdf`
**Authors:** Giovanni De Felice, Arianna Casanova Flores, Francesco De Santis et al. (USI, University of Liechtenstein, Politecnico di Torino, IBM Research)
**Venue:** NeurIPS 2025 (arXiv:2503.04363)

### What It's About

Standard Concept Bottleneck Models (CBMs) force neural networks to reason through human-interpretable concepts (e.g., "this image contains a wheel, therefore it's a car"), but they still learn spurious correlations rather than true causal relationships. This paper proposes Causally Reliable CBMs (C²BMs), which impose a causal graph structure over the concept bottleneck. Information flows through a DAG where each concept is predicted from its causal parents (not directly from all other concepts or raw input). An automated pipeline learns this causal structure from data and unstructured background knowledge (scientific literature).

### Key Contributions

- First framework to enforce causal structure within concept-based neural models.
- Shows that C²BMs are more consistent with real-world causal mechanisms without losing accuracy versus standard CBMs.
- Improves interventional accuracy (e.g., correcting a mispredicted intermediate concept improves downstream accuracy more than in standard CBMs).
- Demonstrates bias mitigation: removing spurious correlations reduces stereotyped predictions.
- Automated causal graph discovery from data + text background knowledge makes deployment practical.

### Limitations

- Requires specifying or learning a causal graph — this is hard in domains where causal structure is poorly understood.
- The hypernetwork used to parameterize structural equations adds model complexity.
- Background knowledge extraction from scientific text is noisy and may introduce errors.
- Evaluated on relatively small tabular/medical datasets; scalability to large language or vision models is unclear.
- Causal graph learning is sensitive to the quality and coverage of background knowledge provided.

### Relevance to Fair-RLVR

This paper provides conceptual grounding for the causal faithfulness experiment (Experiment 4) in Fair-RLVR. The question of whether a model's reasoning chain causally determines its answer — or is post-hoc rationalization — is exactly the interventional causal question that C²BMs are designed to address. Fair-RLVR's faithfulness test (corrupting the CoT and measuring accuracy change) is an informal version of the interventional test studied here.

### Future Directions

- Extending causal bottlenecks to LLM reasoning chains (treating each reasoning step as a "concept").
- Combining causal graph constraints with RL training to encourage causally faithful reasoning.
- Scalable causal discovery for high-dimensional language and vision representations.

---

## 11. Adaptive Safe Context Learning (ASCL)

**File:** `Adaptive_Safe_Context_Learning.pdf`
**Authors:** Yanbo Wang, Minzheng Wang, Jian Liang et al. (Chinese Academy of Sciences, Ritzz-AI)
**Venue:** arXiv:2602.13562 (2026)

### What It's About

ASCL addresses the safety-utility trade-off in Large Reasoning Models (LRMs): safety-aligned models tend to over-refuse benign but superficially sensitive prompts. Standard approaches embed safety rules directly into CoT training data (context distillation), which creates rigid rule-following that hurts utility. ASCL instead decouples safety rules from the reasoning process: the model learns to use safety rules as an external tool that is only invoked when the model decides a query is genuinely risky. This is framed as a multi-turn tool-use problem. Additionally, the paper introduces Inverse Frequency Policy Optimization (IFPO), which rebalances RL advantage estimates to avoid the policy over-consulting safety rules during training.

### Key Contributions

- Decouples safety knowledge from reasoning — safety rules become selectively retrievable context rather than hardcoded memorization.
- Frames safety alignment as multi-turn tool-use: the model decides whether to consult a safety rulebook, then constructs its response given that context.
- IFPO reweights advantages in RL by the inverse frequency of action types, discouraging the model from defaulting to high-frequency actions (like always consulting safety rules).
- Achieves better safety-utility balance than baselines, with lower over-refusal rates on benign prompts.

### Limitations

- Requires a retrieval or tool-use infrastructure to serve safety rules at inference time — adds latency and system complexity.
- The safety rulebook must be manually curated and maintained — doesn't scale automatically to new threat categories.
- IFPO is a heuristic reweighting scheme; theoretical guarantees on its convergence or optimality are not provided.
- Evaluated primarily on Chinese-language safety benchmarks; generalization to other languages is unclear.
- The multi-turn formulation adds inference cost vs. single-turn safety alignment.

### Relevance to Fair-RLVR

ASCL demonstrates that embedding correctness criteria (safety rules) directly into training data is suboptimal — the same argument applies to fairness. Rather than training on examples that say "don't be biased," Fair-RLVR uses a verifiable reward signal that rewards getting the right answer on bias-sensitive questions, allowing the model to discover its own fairness reasoning.

### Future Directions

- Applying adaptive context retrieval to fairness guidelines (e.g., pulling in relevant social context for bias-sensitive questions).
- Combining ASCL's tool-use framework with RLVR for safety to get the benefits of both.
- Evaluating ASCL on English and multilingual safety benchmarks.

---

## 12. Structure Trumps Size: Rethinking Data Quality for LLM Reasoning

**File:** `Structure_Trumps_Size.pdf`
**Authors:** Hu Xu, Zeyan Li, Rui Wang, Jianfeng Xu (Shanghai Jiao Tong University)
**Venue:** EMNLP 2025 Findings (arXiv)

### What It's About

This paper challenges the "more data is better" assumption in SFT for LLM reasoning. Through controlled experiments, it shows that dataset structure — specifically its composition along six measurable dimensions — matters far more than raw size. The paper introduces MCSQ (Multi-dimensional Quantitative Framework for SFT Data Quality), which evaluates datasets along: Volume, Scope (domain breadth), Granularity (reasoning step depth), Variety (diversity of question types), Distortion (model-disagreed samples), and Mismatch (low-relevance samples). Counter-intuitively, including "distorted" or "mismatched" samples that would normally be discarded can boost performance on advanced reasoning benchmarks.

### Key Contributions

- First systematic, quantitative framework (MCSQ) for evaluating reasoning dataset quality along six orthogonal dimensions.
- Shows that data structure consistently beats volume: doubling dataset size beyond a threshold yields diminishing or negative returns.
- Finds that "imperfect" data (samples where model disagrees with ground truth, or samples from adjacent domains) can improve generalization on hard benchmarks.
- Demonstrates that the optimal domain balance (breadth vs. depth) depends on the target task — there is no universal best composition.
- Open-sources datasets and code.

### Limitations

- Experiments are on distilled CoT SFT data — results may not directly transfer to RL-trained models.
- The six dimensions are not fully independent; interactions between dimensions are not modeled.
- "Distorted" samples that help some benchmarks may hurt others — the framework doesn't fully resolve when to include them.
- The study uses relatively small models; scaling behavior of these findings is unclear.

### Relevance to Fair-RLVR

Directly relevant to training data design: Fair-RLVR uses a 0.7/0.3 ambiguous/disambiguated ratio and a 90/10 train/eval split. This paper suggests that the composition of training data (the mix of ambiguous vs. disambiguated, easy vs. hard questions) likely has a larger impact on bias reduction than simply increasing the total training set size.

### Future Directions

- Extending MCSQ to RL training data evaluation.
- Studying the optimal data composition for fairness training specifically.
- Automated dataset curriculum that adaptively adjusts composition as training progresses.

---

## 13. Med-RLVR: Emerging Medical Reasoning via RL

**File:** `Medical_Reasoning_Emergence_via_RL.pdf`
**Authors:** Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, Hoifung Poon (Microsoft Research)
**Venue:** arXiv:2502.19655 (2025)

### What It's About

Med-RLVR is the first paper to apply RLVR (specifically PPO/GRPO with verifiable rewards) to medical multiple-choice question answering. It demonstrates that RLVR is not just for math and coding — it can elicit clinical reasoning from a 3B base model (Qwen2.5-3B) without any explicit reasoning supervision. The paper's other major contribution is a detailed taxonomy of 6 training dynamic phases that emerge during RLVR training. These phases describe how the model's reasoning behavior evolves during training — from chaotic outputs to structured, genuine reasoning. Fair-RLVR directly borrows this 6-phase taxonomy for its training dynamics analysis.

### Key Contributions

- First application of RLVR to medical MCQA, showing the technique generalizes beyond math/coding.
- Achieves performance comparable to SFT on in-distribution MedQA while gaining +8 points on out-of-distribution MMLU-Pro Health — demonstrating better generalization than supervised methods.
- Documents 6 training phases: (1) Unstructured Beginning (fails to follow format), (2) Verbose Formatter (follows format but verbose), (3) Concise Structurer (clear reasoning, good format), (4) Direct Answer Hacker (leaks answer into think block), (5) Step-by-Step Exploit (reasons outside the tags), (6) Reintegrated Reasoning (returns to genuine in-tag reasoning).
- Provides empirical evidence that reasoning emerges from a 3B base model without any CoT supervision.

### Limitations

- Uses PPO rather than GRPO — more memory-intensive and requires a separate value network. Reward hacking behaviors were observed but not mitigated (see Paper 14).
- Evaluated on English medical datasets only; clinical reasoning in other languages is not studied.
- The 6-phase taxonomy is qualitative; precise criteria for phase boundaries are not formalized.
- No safety evaluation — a model that reasons about clinical decisions needs safety constraints.
- Out-of-distribution gains on MMLU-Pro Health are promising but not fully explained mechanistically.

### Relevance to Fair-RLVR

Fair-RLVR's `TrainingDynamicsLogger` in `callbacks.py` directly implements Med-RLVR's 6-phase taxonomy. The training dynamics analysis in the Fair-RLVR paper (Experiment 3) tests whether these same phases appear in fairness training.

### Future Directions

- Combining Med-RLVR with the composite reward function from Paper 14 to reduce reward hacking.
- Extending RLVR to clinical NLP tasks beyond MCQA (radiology report generation, diagnosis explanation).
- Formalizing the 6-phase taxonomy with quantitative phase detection metrics.
- Applying the same RLVR approach to fairness in medical settings (e.g., bias in clinical recommendations).

---

## 14. Reward Hacking Mitigation using Verifiable Composite Rewards

**File:** `Reward_Hacking_Mitigation_VCR.pdf`
**Authors:** Mirza Farhan Bin Tarek, Rahmatollah Beheshti (University of Delaware)
**Venue:** arXiv:2509.15557 (2025)

### What It's About

This paper directly extends Med-RLVR by addressing the reward hacking behaviors it identified but didn't fix. Two specific hacking strategies are observed: (1) the model outputs the correct answer directly inside the `<think>` block, skipping reasoning, to still get a correctness reward; (2) the model reasons in a non-compliant format (e.g., step-by-step reasoning outside the `<think>` tags) to exploit structural ambiguity. The paper introduces a composite reward function with two penalty terms — P_answer (penalizes answer revelation without reasoning) and P_structural (penalizes format violations) — on top of the binary correctness reward. This directly informs Fair-RLVR's structural penalty design.

### Key Contributions

- Formally defines and documents two specific reward hacking failure modes in medical RLVR with concrete examples.
- Introduces a composite reward: R_total = R_binary - P_answer - P_structural.
- P_answer: penalizes outputting the answer inside the `<think>` block (detected via answer token pattern matching).
- P_structural: penalizes format violations including reasoning outside tags, answers revealed before think block, etc.
- Shows the composite reward produces better-formatted reasoning with less hacking and comparable accuracy.
- Evaluates using both automated metrics and human judges.

### Limitations

- Focuses exclusively on medical MCQA — generalization to other domains is not studied.
- P_answer and P_structural are rule-based and hand-crafted — they may miss novel hacking strategies that emerge during longer training.
- No Sentence-BERT or semantic similarity check — purely syntactic penalties may miss semantic leakage (model "hints" at the answer through word choice rather than explicit extraction).
- The penalty weights are not ablated — it's unclear how sensitive results are to the magnitude of each penalty.
- Dataset is small; statistical significance of improvements over Med-RLVR is not rigorously tested.

### Relevance to Fair-RLVR

This is the most direct inspiration for Fair-RLVR's reward function. Fair-RLVR adapts the composite reward idea but simplifies it: it drops the binary R_correctness (since it becomes redundant once format is learned) and the Sentence-BERT penalty (inappropriate for single-letter MCQA answers), keeping only λ·R_fairness - P_structural. The P_structural in Fair-RLVR checks the same three conditions: answer leaked in think, reasoning too short, content outside tags.

### Future Directions

- Semantic leakage detection using embedding similarity (as originally attempted in Fair-RLVR but removed for MCQA).
- Adaptive penalty weights that increase as training progresses (starting lenient, becoming strict).
- Applying composite rewards to fairness training — the core contribution of Fair-RLVR.
- Combining with process reward models to reward individual reasoning steps rather than just final format.

---

## 15. Mitigating Bias in RLHF for Large Language Models

**File:** `Mitigating_Bias_in_Reinforcement_Learning_from_Human_Feedback_for_Large_Language_Models.pdf`
**Authors:** Chaithanya Ravulu (Rocket Companies); Rahul Sarabu, Manoj Suryadevara, Mridula Dileepraj Kidiyur (Walmart); Venkata Gummadi (Expedia)
**Venue:** 2024 IEEE International Conference on AI x Data and Knowledge Engineering (AIxDKE) — DOI 10.1109/AIxDKE63520.2024.00019

### What It's About

A practitioner-oriented empirical comparison of four bias-mitigation strategies layered on top of standard RLHF, tested on sentiment classification (BERT) and three GPT-3-style tasks (text completion, QA, dialogue generation):

1. **Diverse Feedback (DF)** — demographically balanced annotator panel.
2. **Bias Correction (BC)** — two stages. Reward debiasing subtracts an evaluator-bias term: r̂_i = r_i − b_i(x). Output debiasing reshapes the policy via a Boltzmann-style reweight: π_debiased(a|s) = π_θ(a|s) · exp(−λB(a,s)) / Z(s).
3. **Fairness Constraints (FC)** — Lagrangian augmentation of the RLHF objective: J_fair(θ) = J(θ) − Σ_i α_i C_i(θ), where C_i are demographic-parity / equal-opportunity terms.
4. **Counterfactual Data Augmentation (CDA)** — swap protected attributes in training prompts, train on the union, add a reward bonus for consistent answers across counterfactual pairs.

Reported against four fairness metrics — Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD), Disparate Impact Ratio (DIR), Representation Bias (RB) — plus task-specific accuracy/perplexity/F1/BLEU.

### Key Contributions

- Catalogues four orthogonal mitigation axes (data diversity, reward correction, constrained optimization, counterfactual augmentation) inside one framework — a useful taxonomy even though each idea exists in prior literature.
- Reports stacking all four ("ALL") is super-additive: 60–70% bias reduction across tasks vs. 30–60% for any single method.
- Among singletons: FC gives the largest bias drop but the worst accuracy hit (3.4%); BC is the best accuracy/fairness trade-off; CDA is the most consistent across metrics.
- Paired t-tests across seeds: bias improvements significant at p<0.01 for all methods; performance drops only consistently significant for FC and ALL — suggesting DF/BC/CDA can be added with low risk to task quality.

### Limitations

- **Uses RLHF, ignores known structural pathologies.** Papers 6 (Sycophancy Amplification) and 9 (Preference Collapse) already prove RLHF amplifies annotator bias and suppresses minority preferences. This paper proposes patches but never acknowledges the underlying issue — its mitigations are bandaids on a structurally biased optimizer.
- **The reward debiasing term b_i(x) is unspecified.** The text doesn't say how the evaluator-bias function is estimated, validated, or kept calibrated as evaluator pools shift.
- **Results tables look suspiciously monotonic.** Every metric improves uniformly under every method, which is rare in real fairness work — DPD and EOD typically trade off in practice.
- **No standardized benchmark.** Bias is measured with abstract DPD/EOD/DIR on internal datasets — never BBQ, StereoSet, or WinoBias. Hard to compare against published baselines.
- **No causal analysis.** Whether the model's behavior is *because* of fairness reasoning, or whether constraints just suppress generation, is never investigated.
- **English only, BERT-era.** Sentiment uses BERT (encoder-only); QA/dialogue use unspecified GPT-3 style. Generalization to modern decoder-only reasoning LLMs is untested.
- **CDA assumes well-defined attribute swaps.** Race/gender swaps are easy; SES, disability, religion, intersectional are not. The paper sidesteps this.

### Relevance to Fair-RLVR

This is an RLHF-camp paper; Fair-RLVR is RLVR-camp. But three of the four techniques transfer cleanly to RLVR with objective labels:

- **CDA → counterfactual-consistency reward.** BBQ templates already include demographic fills (the same template instantiated with different names/groups). This dataset structure is currently unused at training time — the natural Fair-RLVR extension is a consistency term rewarding the same answer across (prompt_v1, prompt_v2) when only the demographic varies.
- **FC → group-disparity penalty in GRPO.** The Lagrangian J_fair = J − Σ α_i C_i(θ) maps directly to adding a per-batch term `−α · (max_cat acc − min_cat acc)` to the reward, enforcing parity across BBQ's 9 categories.
- **DF → curriculum / weighted sampling.** "Diverse feedback" doesn't apply (no annotators), but its analog is reweighting the BBQ training distribution toward currently weakest categories.
- **BC** has no clean RLVR analog — there's no annotator to debias because the reward is automated. This is a feature of RLVR, not a missing capability.

The paper also exposes a gap in our planned evaluation: we report BBQ-official bias score and the simplified error-based metric, but not DPD/EOD/DIR/RB. Adding these would make our numbers comparable to the broader fairness literature.

### Future Directions

- Replicating the FC + CDA + DF stack inside RLVR (instead of RLHF) to verify the "stacking is super-additive" claim under verifiable reward.
- Causal/faithfulness analysis of which mitigation actually shifts reasoning vs. just suppresses outputs (this paper does none; Fair-RLVR Experiment 4 partly fills this gap).
- Standardizing on shared benchmarks (BBQ, StereoSet, WinoBias) so mitigation methods can be cross-paper compared.
- Extending counterfactual swaps to non-binary attributes (intersectional categories, SES, disability).

---

## Quick Comparison Table

| Paper | Core Contribution | Relevance to Fair-RLVR |
|---|---|---|
| DeepSeek-R1 | RLVR elicits emergent reasoning via pure RL | Foundational paradigm; GRPO + format reward |
| DeepSeekMath | Introduces GRPO algorithm | GRPO is Fair-RLVR's training algorithm |
| BBQ | Bias benchmark with objective labels | The fairness verifier in Fair-RLVR |
| FairReason | RL best for bias+reasoning balance; 1:4 data ratio | Validates RL approach; informs λ sweep |
| RealSafe-R1 | Safety alignment via CoT distillation | Shows SFT can align reasoning models |
| RLHF Amplifies Sycophancy | RLHF formally amplifies human bias | Justifies abandoning RLHF for fairness |
| DAPO | 4 GRPO fixes; asymmetric clipping | Fair-RLVR uses DAPO clipping ratios |
| VMR-RLVR | MC reformulation extends RLVR to any domain | Validates using MCQA as verifiable signal |
| RLHF Preference Collapse | KL regularization suppresses minorities | More evidence RLHF fails for fairness |
| C²BMs | Causal structure in concept models | Motivates causal faithfulness experiment |
| ASCL | Adaptive safety rule retrieval; IFPO | Contrasts with Fair-RLVR's reward approach |
| Structure Trumps Size | Data composition > data volume | Informs training data ratio design |
| Med-RLVR | RLVR for medical MCQA; 6 training phases | Source of 6-phase taxonomy in Fair-RLVR |
| VCR Reward Hacking | Composite reward stops reward hacking | Direct inspiration for P_structural penalty |
| Mitigating Bias in RLHF | DF / BC / FC / CDA stack on RLHF | Three techniques (CDA, FC, DF) transfer to RLVR; motivates new ablations + DPD/EOD/DIR metrics |
