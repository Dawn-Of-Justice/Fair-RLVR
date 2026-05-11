# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) to **fairness alignment** using the BBQ benchmark as an automated fairness verifier. The model (Qwen2.5-3B-Instruct + LoRA) is trained via GRPO with a simplified two-component composite reward:

```
R_total = λ · R_fairness + α · R_consistency - P_structural
```

The model must generate outputs in `<think>...</think><answer>(a/b/c)</answer>` format. R_correctness and P_leak were removed from the original 4-component design; R_consistency was added as a counterfactual-consistency bonus (Ravulu et al. 2024 CDA, RLVR-adapted) — see Key Design Decisions for rationale. α defaults to 0 (off) in the lambda sweep and 0.25 in the main `fair_rlvr.yaml` run.

## Commands

```bash
# Install
pip install -e .
pip install -r requirements.txt

# Dry run (5 steps, verifies pipeline works end-to-end)
python -m src.train --dry-run

# Run baselines
python -m src.baselines.baseline_model --seed 42   # zero-shot
python -m src.baselines.sft --seed 42              # supervised fine-tune
python -m src.baselines.grpo_no_fairness --seed 42 # GRPO λ=0 ablation

# Lambda sweep — λ=0.5 run doubles as the main experiment
python -m src.train --config configs/lambda_sweep.yaml --lambda-fair 0.1 --output-dir results/lambda_0.1
python -m src.train --config configs/lambda_sweep.yaml --lambda-fair 0.3 --output-dir results/lambda_0.3
python -m src.train --config configs/lambda_sweep.yaml --lambda-fair 0.5 --output-dir results/lambda_0.5
python -m src.train --config configs/lambda_sweep.yaml --lambda-fair 0.7 --output-dir results/lambda_0.7
python -m src.train --config configs/lambda_sweep.yaml --lambda-fair 1.0 --output-dir results/lambda_1.0

# Train with Weights & Biases monitoring
python -m src.train --config configs/fair_rlvr.yaml --wandb --wandb-project fair-rlvr

# Evaluate each lambda checkpoint (full eval split by default)
python -m src.evaluate --checkpoint results/lambda_0.5/final_adapter  # main result

# Interventional sensitivity test on main model (Experiment 4 — three-level)
python -m src.evaluate --checkpoint results/lambda_0.5/final_adapter --run-faithfulness

# OOD evaluation on WinoBias + StereoSet (Experiment 5)
python -m src.evaluate --checkpoint results/lambda_0.5/final_adapter --run-ood

# Run everything at once
python -m src.evaluate --checkpoint results/lambda_0.5/final_adapter --run-faithfulness --run-ood

# Test reward function directly
python -m src.reward

# Test data pipeline (confirms template-family-aware split sizes)
python -m src.data
```

## Architecture

### Data flow (`src/data.py`)
- Loads BBQ from HuggingFace (`Elfsong/BBQ`) across all 9 bias categories
- `create_splits()` performs a **template-family-aware 90/10 split** — groups examples by `(category, question_index)` so ambiguous/disambiguated pairs and demographic variants always land in the same split (prevents near-duplicate leakage)
- Training set: ~52,643 samples. Eval set: ~5,849 samples. Seed=42 must be consistent across all scripts.
- **All 9 categories are in the training split** — there are no held-out BBQ categories. OOD evaluation uses WinoBias, StereoSet, and intersectional BBQ categories (`race_x_gender`, `race_x_ses`) loaded separately via `load_bbq_intersectional()`
- Each example gets an `unknown_label` field (index 0/1/2) via `get_unknown_label()` — the "Unknown" option is not always option (c); its position varies per question
- `SYSTEM_PROMPT` in this file is the canonical prompt used across all training and inference
- `sort_by_family=True` sorts training indices by `question_index` so demographic siblings land in adjacent positions, enabling the R_consistency sibling lookup in `make_reward_fn()`

### Reward function (`src/reward.py`)
- `compute_reward()` is the core function — returns a dict with all reward components
- `reward_fairness()` extracts the `<answer>` tag, maps `(a)/(b)/(c)` to index 0/1/2, returns +1.0 if it matches BBQ ground truth label, else 0.0
- `reward_consistency()` returns +1.0 if the model's answer matches ALL demographic sibling outputs from the same BBQ template family (Ravulu et al. 2024 CDA adapted to RLVR). Off by default (α=0).
- `penalty_structural()` checks four rule-based violations, each costing 0.3 (max penalty = 1.2):
  1. Reasoning too short or `<think>` tag missing (< 20 tokens)
  2. Missing `<answer>` tag entirely
  3. Answer leaked into `<think>` block
  4. Content exists outside `<think>`/`<answer>` tags
- No SBERT dependency — `P_leak` (Sentence-BERT cosine similarity) was removed; single-letter MCQA answers `(a)/(b)/(c)` never reach the τ=0.85 threshold against multi-sentence reasoning chains

### Training (`src/train.py`)
- `make_reward_fn()` wraps `compute_reward` into TRL's GRPOTrainer signature (takes `completions`, `**kwargs` with `prompts`)
- `build_grpo_dataset()` formats BBQ into a HuggingFace `Dataset` with a `"prompt"` string column; returns both a `ground_truth_map` dict (formatted prompt → answer label) and a `family_map` dict (formatted prompt → `(category, question_index)`) for sibling lookup
- Uses DAPO-style asymmetric clipping: `epsilon=clip_ratio_low=0.20`, `epsilon_high=clip_ratio_high=0.28` — **actually passed to GRPOConfig** (requires TRL ≥0.12)
- Weights & Biases: pass `--wandb` to enable. TRL logs loss/reward/KL; `FairRLVRCallback` additionally logs reward breakdown and abstention rate per batch.
- Config files override defaults; CLI args override config files

### Evaluation (`src/evaluate.py`)
- `run_evaluation()` loads a LoRA adapter on top of the base model (no quantization at eval time — uses float16)
- **Primary bias metric**: Official BBQ bias score matching `BBQ_calculate_bias_score.R` exactly, implemented in `compute_bbq_official_bias_score()`:
  1. Filter "Unknown" predictions from denominator (using per-question `unknown_label` index): `raw = 2 × P(target | prediction ≠ Unknown) − 1`
  2. Scale ambiguous score by accuracy: `bias_bbq_ambig = raw × (1 − accuracy_ambig)`
  3. Disambiguated score is unscaled: `bias_bbq_disambig = raw`
  Range [−1, 1]; 0 = unbiased; positive = stereotype-aligned.
- **Secondary bias metric**: Stereotype-consistent errors / total errors (range [0, 1]; 0.5 = unbiased)
- Abstention rate uses `unknown_label` field for per-question index comparison — string heuristics and hardcoded index=2 are both wrong
- **OOD evaluation** (`--run-ood`): `evaluate_winobias()` reformats Type-2 WinoBias as 2-choice MC QA, reports pro vs anti-stereotypical accuracy gap. `evaluate_stereoset()` reformats StereoSet intrasentence as 3-choice MC, reports Stereotype Score (SS), Language Model Score (LMS), and ICAT.
- `compute_faithfulness()` (Experiment 4) runs a **three-level interventional test**:
  - Condition A (real CoT): baseline — always 1.0 since we filter to correct predictions
  - Condition B (permuted CoT): sentences shuffled — tests word-order sensitivity
  - Condition C (null CoT): `[No reasoning provided]` — tests whether CoT *content* is needed at all
  - `sensitivity_permuted = P(A) − P(B)`, `sensitivity_null = P(A) − P(C)`
  - **Interpretation**: if `sensitivity_null ≈ 0`, de-biasing is driven by prompt-level representations, not the reasoning chain. Do not claim CoT is causally necessary in this case — report the result and qualify.

### Callbacks (`src/callbacks.py`)
- `FairRLVRCallback`: logs reward mean/std per step via `on_step_end`; saves CoT samples at configurable checkpoint steps (proportionally spaced at 3%, 14%, 29%, 43%, 57%, 71%, 86%, 100% of `num_train_steps`); logs W&B metrics when `use_wandb=True`
- `TrainingDynamicsLogger`: classifies each batch into one of 6 training phases (Format Failure → Reintegrated Reasoning) tracking reward hacking progression; logs phase metrics to W&B when enabled
- `log_generation_batch()` is called automatically from inside `make_reward_fn()` in `train.py` — logs `avg_r_fairness`, `avg_r_consistency`, `avg_p_structural`, `batch_accuracy`, `batch_abstention_rate` per batch

## Key Design Decisions

- **Reward equation**: `R_total = λ · R_fairness + α · R_consistency - P_structural`
  - `R_correctness` removed: redundant once `P_structural` covers format violations; zeros out in GRPO group normalization once format is stable
  - `P_leak` (Sentence-BERT) removed: single-letter MCQA answers `(a)/(b)/(c)` never reach cosine similarity τ=0.85 against multi-sentence reasoning chains — dead code
  - `R_consistency` added: counterfactual-consistency bonus (Ravulu et al. 2024 CDA, RLVR-adapted). +1 if the predicted ANSWER TEXT (option content, not the (a)/(b)/(c) index) matches an in-batch sibling from the same BBQ template family but a different demographic fill. Off by default (α=0); set `alpha_consistency: 0.25` in the YAML or pass `--alpha-consistency 0.25` to enable.
- **Lambda (`λ`)**: Controls fairness signal strength. Even `λ=0.1` achieves debiasing. Default is 0.5. Max reward at λ=0.5, α=0 is `0.5`; with α=0.25 the max becomes `0.75`. Reward range is `[−1.2, λ + α]`.
- **Training uses 4-bit quantization** (`bitsandbytes` nf4); evaluation loads in float16 without quantization.
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all attention + MLP projections). All baselines use the same targets for equal parameter counts.
- **Group size G=2**: For 3-choice MCQA the reward signal is dominated by correctness (3 possible outcomes), so the GRPO group baseline is well-estimated even at G=2. Lower G drastically reduces generation cost vs the standard G=8; if entropy collapses, raise G first.
- **Template-family-aware split**: A naive `train_test_split` stratified only by category puts near-duplicate question variants across splits, inflating eval scores. `create_splits()` groups by `(category, question_index)` to prevent this.
- **`unknown_label` field**: The "Unknown" answer is not always option (c). Always use the per-question `unknown_label` index for abstention detection and the official bias score denominator.
- **Seed consistency**: All scripts (train, eval, all baselines) must use `--seed 42` (the default). Mismatched seeds produce different train/eval splits → data leakage.
- **DAPO asymmetric clipping**: `epsilon=0.20` (low) and `epsilon_high=0.28` (high) are **passed to GRPOConfig** (requires TRL ≥0.12). Clip-Higher promotes diversity and prevents entropy collapse without penalizing high-probability correct outputs.
- **Sibling co-batching**: When `alpha_consistency > 0`, `create_splits(sort_by_family=True)` sorts by `question_index` so demographic variants of the same BBQ template land in adjacent positions. `make_reward_fn()` groups prompts by `family_map` within each batch to find siblings.
- The `ground_truth_map` keyed on formatted prompt strings is the mechanism tying GRPO completions back to their ground-truth labels — rebuilt from scratch each run. A warning is printed if any prompt has no ground truth (silent failures were a bug in the original code).

## OOD Generalization (Experiment 5) — Important Scope Limitation

**Do NOT claim generalization to "held-out BBQ categories"** — all 9 base categories are in training. The benchmark overfitting concern (all λ values perform similarly, model may leverage pre-training familiarity with BBQ-style contexts) is addressed by evaluating on:

- **WinoBias** (`uclanlp/winobias`): Gender/coreference bias. Type-2 sentences are syntactically unambiguous — the correct referent is clear from syntax. Bias score = acc_pro − acc_anti; 0 = unbiased.
- **StereoSet** (`McGill-NLP/stereoset`): Association bias via intrasentence MC. Stereotype Score (SS=0.5 = unbiased), Language Model Score, ICAT.
- **Intersectional BBQ** (`race_x_gender`, `race_x_ses`): Loaded separately via `load_bbq_intersectional()`.

## Causal Faithfulness — Interpretation Constraint

The three-level faithfulness test (Experiment 4) may show P(correct | null CoT) ≈ P(correct | real CoT) ≈ 1.0. This means:
- De-biasing behavior is primarily driven by **prompt-level representations** learned during RLVR training
- The chain-of-thought may be **post-hoc rationalization**, not a causal mechanism
- Do NOT claim the model "learns to identify and reject stereotypes as logical errors within the CoT" if `sensitivity_null < 0.05`
- Instead, report: de-biasing is internalized at the representation level; the CoT is not causally necessary but remains a useful output format for interpretability

The `_faithfulness_interpretation()` function in `evaluate.py` generates a standardized flag (`HIGH_NULL_ACCURACY` / `MODERATE_NULL_ACCURACY` / `LOW_NULL_ACCURACY`) to enforce consistent reporting.

## Baselines

| Baseline | Script | What it isolates |
|---|---|---|
| Zero-shot | `baselines/baseline_model.py` | Qwen2.5-3B-Instruct out-of-the-box fairness |
| SFT | `baselines/sft.py` | Supervised learning on BBQ labels (bfloat16, 7 LoRA targets) |
| GRPO λ=0 | `baselines/grpo_no_fairness.py` | `R_total = −P_structural` only — isolates whether GRPO format training alone drives debiasing |

The λ=0 baseline is the critical ablation: if it approaches Fair-RLVR's bias score, GRPO structure explains the result; if it stays near 0 (official metric), the explicit fairness reward is the mechanism.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 4-bit bitsandbytes nf4 (training); float16 (eval) |
| LoRA r / α | 16 / 32 |
| Group size G | 2 |
| Batch size | 8 per device (grad_accum=2 → effective batch=16) |
| Max new tokens | 512 |
| Learning rate | 1e-5 |
| KL coefficient | 0.01 |
| Clip ratios | ε_low=0.20, ε_high=0.28 (DAPO asymmetric, actually wired to GRPOConfig) |
| Training steps | 3,500 |
| Checkpoint cadence | Every 500 steps |
| Dataset | Full BBQ, 90/10 template-family split, seed=42 |

## Experiments

- **Exp 1 (Main)**: Zero-shot vs SFT vs GRPO λ=0 vs Fair-RLVR λ=0.5. Reports BBQ accuracy (ambig/disambig), official bias score (primary), secondary bias score, MMLU, GSM8K.
- **Exp 2 (Lambda sweep)**: λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}. Watch for λ>0.7 → over-abstention; λ<0.3 → fairness signal lost.
- **Exp 3 (Training dynamics)**: 6-phase Med-RLVR taxonomy applied to fairness training. Logged by `TrainingDynamicsLogger`; CoT samples saved at 8 proportionally-spaced checkpoints.
- **Exp 4 (Interventional sensitivity)**: Three-level CoT sufficiency test. Reports `sensitivity_permuted` and `sensitivity_null`. If null sensitivity is near 0, qualify causal claims (see Causal Faithfulness section above).
- **Exp 5 (OOD generalization)**: WinoBias + StereoSet + intersectional BBQ. Addresses benchmark overfitting concern. Run with `--run-ood`.
- **Exp 6 (Bias amplification check)**: Track official bias score at each 500-step checkpoint. FairReason found 3B models can get *more* biased at intermediate steps — document if/where this occurs.

## Weights & Biases Monitoring

Enable with `--wandb` (or `use_wandb: true` in config). Metrics logged:

| Metric | Source | W&B key |
|---|---|---|
| Loss, LR, reward mean/std, KL | TRL (automatic) | `train/loss`, `train/reward`, etc. |
| R_fairness avg per batch | FairRLVRCallback | `train/avg_r_fairness` |
| R_consistency avg per batch | FairRLVRCallback | `train/avg_r_consistency` |
| P_structural avg per batch | FairRLVRCallback | `train/avg_p_structural` |
| Batch accuracy | FairRLVRCallback | `train/batch_accuracy` |
| Batch abstention rate | FairRLVRCallback | `train/batch_abstention_rate` |
| Dominant training phase (1–6) | TrainingDynamicsLogger | `dynamics/dominant_phase` |
| Real reasoning fraction | TrainingDynamicsLogger | `dynamics/real_reasoning_frac` |
| Hacker fraction | TrainingDynamicsLogger | `dynamics/hacker_frac` |

## Output Structure

Each experiment writes to `results/<name>/`:
- `config.json` — hyperparameters used
- `logs/step_logs.json` — per-step reward metrics
- `logs/cot_samples.json` — CoT examples at checkpoint steps
- `dynamics/phase_log.json` — training phase classifications
- `final_adapter/` — LoRA adapter weights + tokenizer
- `metrics.json` — evaluation results (BBQ + OOD if run)
- `ood_metrics.json` — WinoBias + StereoSet results (if `--run-ood`)
- `predictions.json` — per-sample model outputs
