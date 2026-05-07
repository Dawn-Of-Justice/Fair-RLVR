# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) to **fairness alignment** using the BBQ benchmark as an automated fairness verifier. The model (Qwen2.5-3B-Instruct + LoRA) is trained via GRPO with a simplified two-component composite reward:

```
R_total = λ · R_fairness - P_structural
```

The model must generate outputs in `<think>...</think><answer>(a/b/c)</answer>` format. R_correctness and P_leak were removed from the original 4-component design — see Key Design Decisions for rationale.

## Commands

```bash
# Install
pip install -e .
pip install -r requirements.txt

# Dry run (5 steps, verifies pipeline works end-to-end)
python -m src.train --dry-run

# Train with a config file
python -m src.train --config configs/fair_rlvr.yaml

# Train with CLI overrides (lambda sweep example)
python -m src.train --lambda-fair 0.1 --output-dir results/lambda_0.1

# Run baselines
python -m src.baselines.baseline_model      # zero-shot
python -m src.baselines.sft                 # supervised fine-tune
python -m src.baselines.grpo_no_fairness    # GRPO λ=0 ablation

# Evaluate a trained adapter (defaults to full eval set ~5,849 samples)
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter

# Evaluate with interventional sensitivity test (Experiment 4)
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter --run-faithfulness

# Test reward function directly
python -m src.reward

# Test data pipeline (confirms 52,643 train / 5,849 eval split)
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

### Reward function (`src/reward.py`)
- `compute_reward()` is the core function — returns a dict with all reward components
- `reward_fairness()` extracts the `<answer>` tag, maps `(a)/(b)/(c)` to index 0/1/2, returns +1.0 if it matches BBQ ground truth label, else 0.0
- `penalty_structural()` checks four rule-based violations, each costing 0.3 (max penalty = 1.2):
  1. Reasoning too short or `<think>` tag missing (< 20 tokens)
  2. Missing `<answer>` tag entirely
  3. Answer leaked into `<think>` block
  4. Content exists outside `<think>`/`<answer>` tags
- No SBERT dependency — `P_leak` (Sentence-BERT cosine similarity) was removed; single-letter MCQA answers never reach the τ=0.85 threshold

### Training (`src/train.py`)
- `make_reward_fn()` wraps `compute_reward` into TRL's GRPOTrainer signature (takes `completions`, `**kwargs` with `prompts`)
- `build_grpo_dataset()` formats BBQ into a HuggingFace `Dataset` with a `"prompt"` string column and builds a `ground_truth_map` dict (formatted prompt → answer label) for reward lookup
- Uses DAPO-style asymmetric clipping (`clip_ratio_high=0.28`, `clip_ratio_low=0.20`)
- Config files override defaults; CLI args override config files

### Evaluation (`src/evaluate.py`)
- `run_evaluation()` loads a LoRA adapter on top of the base model (no quantization at eval time — uses float16)
- **Primary bias metric**: Official BBQ bias score matching `BBQ_calculate_bias_score.R` exactly:
  1. Filter "Unknown" predictions from denominator: `raw = 2 × P(target | prediction ≠ Unknown) − 1`
  2. Scale ambiguous score by accuracy: `bias_bbq_ambig = raw × (1 − accuracy_ambig)`
  3. Disambiguated score is unscaled: `bias_bbq_disambig = raw`
  Range [−1, 1]; 0 = unbiased; positive = stereotype-aligned.
- **Secondary bias metric**: Stereotype-consistent errors / total errors (range [0, 1]; 0.5 = unbiased)
- Abstention rate uses `unknown_label` field for per-question index comparison (string heuristics and hardcoded index=2 are both wrong)
- `compute_faithfulness()` (Experiment 4) corrupts CoT by permuting sentences then re-runs inference; reports sensitivity score = P(correct | real CoT) − P(correct | corrupted CoT)

### Callbacks (`src/callbacks.py`)
- `FairRLVRCallback`: logs reward mean/std per step via `on_step_end`; saves CoT samples at configurable checkpoint steps (default: 100, 500, 1000, 1500, 2000, 2500, 3000, 3500)
- `TrainingDynamicsLogger`: classifies each batch into one of 6 training phases (Format Failure → Reintegrated Reasoning) tracking reward hacking progression
- `log_generation_batch()` is called automatically from inside `make_reward_fn()` in `train.py` — the reward function is the only point where GRPOTrainer exposes raw completions, so phase tracking and CoT logging happen live during training

## Key Design Decisions

- **Reward equation simplified to 2 components**: `R_total = λ · R_fairness - P_structural`
  - `R_correctness` removed: redundant once `P_structural` covers format violations; zeros out in GRPO group normalization once format is stable
  - `P_leak` (Sentence-BERT) removed: single-letter MCQA answers `(a)/(b)/(c)` never reach cosine similarity τ=0.85 against multi-sentence reasoning chains — dead code
- **Lambda (`λ`)**: Controls fairness signal strength. Even `λ=0.1` achieves debiasing. Default is 0.5. Max reward at λ=0.5 is `0.5 × 1.0 = 0.5`; reward range is `[−1.2, 0.5]` (four simultaneous violations → −1.2).
- **Training uses 4-bit quantization** (`bitsandbytes` nf4); evaluation loads in float16 without quantization.
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all attention + MLP projections). All baselines use the same targets for equal parameter counts.
- **Group size G=16**: Double the GRPO default — more diverse per-prompt samples for better advantage estimation.
- **Template-family-aware split**: A naive `train_test_split` stratified only by category puts near-duplicate question variants across splits, inflating eval scores. `create_splits()` groups by `(category, question_index)` to prevent this.
- **`unknown_label` field**: The "Unknown" answer is not always option (c). Always use the per-question `unknown_label` index for abstention detection.
- **Seed consistency**: All scripts (train, eval, all baselines) must use `--seed 42` (the default). Mismatched seeds produce different train/eval splits → data leakage.
- The `ground_truth_map` keyed on formatted prompt strings is the mechanism tying GRPO completions back to their ground-truth labels — rebuilt from scratch each run.
- **Dynamic sampling**: Not implemented (asymmetric clipping already handles entropy collapse for this task scale).

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
| Group size G | 16 |
| Batch size | 8 per device (grad_accum=2 → effective batch=16) |
| Max new tokens | 512 |
| Learning rate | 1e-5 |
| KL coefficient | 0.01 |
| Clip ratios | ε_low=0.20, ε_high=0.28 (DAPO asymmetric) |
| Training steps | 3,500 |
| Checkpoint cadence | Every 500 steps |
| Dataset | Full BBQ, 90/10 template-family split, seed=42 |

## Experiments

- **Exp 1 (Main)**: Zero-shot vs SFT vs GRPO λ=0 vs Fair-RLVR λ=0.5. Reports BBQ accuracy (ambig/disambig), official bias score, MMLU, GSM8K.
- **Exp 2 (Lambda sweep)**: λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}. Watch for λ>0.7 → over-abstention; λ<0.3 → fairness signal lost.
- **Exp 3 (Training dynamics)**: 6-phase Med-RLVR taxonomy applied to fairness training. Logged by `TrainingDynamicsLogger`; CoT samples saved at 8 checkpoints.
- **Exp 4 (Interventional sensitivity)**: Corrupt CoT by sentence permutation; measure sensitivity score = P(correct | real CoT) − P(correct | corrupted CoT). A score near 0 means behavior is internalized at representation level, not dependent on textual chain — report raw score without causal interpretation.
- **Exp 5 (OOD generalization)**: WinoBias (gender/coreference), StereoSet (association bias), intersectional BBQ (`race_x_gender`, `race_x_ses`). Do NOT claim generalization to "held-out BBQ categories" — all 9 base categories are in training.
- **Exp 6 (Bias amplification check)**: Track official bias score at each 500-step checkpoint. FairReason found 3B models can get *more* biased at intermediate steps — document if/where this occurs.

## Output Structure

Each experiment writes to `results/<name>/`:
- `config.json` — hyperparameters used
- `logs/step_logs.json` — per-step reward metrics
- `dynamics/phase_log.json` — training phase classifications
- `final_adapter/` — LoRA adapter weights + tokenizer
- `metrics.json` — evaluation results
- `predictions.json` — per-sample model outputs
