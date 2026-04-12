# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) to **fairness alignment** using the BBQ benchmark as an automated fairness verifier. The model (Qwen2.5-3B-Instruct + LoRA) is trained via GRPO with a composite reward:

```
R_total = R_correctness + λ · R_fairness - P_structural - P_leak
```

The model must generate outputs in `<think>...</think><answer>(a/b/c)</answer>` format to receive any reward.

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
python -m src.baselines.zero_shot --n-eval 500
python -m src.baselines.sft --n-train 1000 --n-eval 500

# Evaluate a trained adapter
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter

# Evaluate with causal faithfulness test (Experiment 4)
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter --run-faithfulness

# Test reward function directly
python -m src.reward

# Test data pipeline
python -m src.data
```

## Architecture

### Data flow (`src/data.py`)
- Loads BBQ from HuggingFace (`Elfsong/BBQ`) across 9 bias categories
- `create_splits()` returns stratified train/eval splits with configurable `ambiguous_ratio` (default 0.7 ambig : 0.3 disambig)
- Religion and sexual_orientation are held out by default as OOD eval categories
- Intersectional categories (`race_x_gender`, `race_x_ses`) are always held out separately
- `SYSTEM_PROMPT` in this file is the canonical prompt used across all training and inference

### Reward function (`src/reward.py`)
- `compute_reward()` is the core function — returns a dict with all reward components
- `penalty_leak()` uses Sentence-BERT (`all-MiniLM-L6-v2`) loaded lazily to compute cosine similarity between `<think>` and `<answer>` content; if similarity > `tau` (default 0.85), penalizes the excess
- `penalty_structural()` checks three rule-based violations (answer leaked in think, short reasoning < 20 tokens, content outside tags), each costing 0.3
- The SBERT model is a module-level singleton (`_sbert_model`) — avoid re-importing in hot paths

### Training (`src/train.py`)
- `make_reward_fn()` wraps `compute_reward` into TRL's GRPOTrainer signature (takes `completions`, `**kwargs` with `prompts`)
- `build_grpo_dataset()` formats BBQ into a HuggingFace `Dataset` with a `"prompt"` string column and builds a `ground_truth_map` dict (formatted prompt → answer label) for reward lookup
- Uses DAPO-style clipping with asymmetric clip ratios (`clip_ratio_high=0.28`, `clip_ratio_low=0.20`)
- Config files override defaults; CLI args override config files

### Evaluation (`src/evaluate.py`)
- `run_evaluation()` loads a LoRA adapter on top of the base model (no quantization at eval time — uses float16)
- `compute_bias_score()` measures proportion of errors that are stereotype-consistent using BBQ's `target_label` field
- `compute_faithfulness()` (Experiment 4) corrupts CoT by permuting sentences then re-runs inference to measure causal link

### Callbacks (`src/callbacks.py`)
- `FairRLVRCallback`: logs reward stats per step, saves CoT samples at configurable checkpoint steps
- `TrainingDynamicsLogger`: classifies each batch into one of 6 training phases (Format Failure → Reintegrated Reasoning) tracking reward hacking progression

## Key Design Decisions

- **Lambda (`λ`)**: Controls fairness-vs-correctness tradeoff. Even `λ=0.1` achieves strong debiasing. Default is 0.5.
- **Training uses 4-bit quantization** (`bitsandbytes` nf4); evaluation loads in float16 without quantization.
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all attention + MLP projections).
- **Bias score < 0.5** means fair; 0.5 = random errors; > 0.5 = stereotyped errors.
- The `ground_truth_map` keyed on formatted prompt strings is the mechanism tying GRPO completions back to their ground-truth labels — this is rebuilt from scratch each training run.

## Detailed Results (from paper)

**Per-category bias scores** (Zero-shot → Fair-RLVR):
| Category | Zero-shot | Fair-RLVR |
|---|---|---|
| Age | 0.727 | 0.316 |
| SES | 0.655 | 0.500 |
| Nationality | 0.556 | 0.111 |
| Physical Appearance | 0.571 | 0.400 |
| Race/Ethnicity | 0.476 | 0.231 |
| Gender Identity | 0.304 | **0.000** |
| Disability | 0.500 | 0.500 |

**Training dynamics**: Mean reward rose from 1.406 (step 11) → 1.500 (step 500, theoretical max for λ=0.5). Reward std collapsed to 0.0 at mid-training, then re-emerged at step 1000 (mean 1.472, std 0.026). KL stabilized at 0.017.

**Causal faithfulness test** (100 samples, sentence permutation corruption): P(correct | real CoT) = 1.000, P(correct | corrupted CoT) = 0.990, faithfulness score = **0.010** — fair behavior is internalized at the representation level, not dependent on the textual reasoning chain.

## Output Structure

Each experiment writes to `results/<name>/`:
- `config.json` — hyperparameters used
- `logs/step_logs.json` — per-step reward metrics
- `dynamics/phase_log.json` — training phase classifications
- `final_adapter/` — LoRA adapter weights + tokenizer
- `metrics.json` — evaluation results
- `predictions.json` — per-sample model outputs
