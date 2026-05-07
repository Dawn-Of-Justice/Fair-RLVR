# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Summary

Fair-RLVR studies whether fairness behavior can be trained with Reinforcement Learning
from Verifiable Rewards (RLVR) using BBQ as the verifier. The implemented setup uses:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Training method: GRPO with LoRA
- Task format: BBQ multiple-choice QA
- Required output format:
  `<think>...</think><answer>(a|b|c)</answer>`

The current implemented reward is:

```text
R_total = lambda_fair * R_fairness - P_structural
```

Do not reintroduce `R_correctness` or `P_leak` unless the user explicitly asks for a
design change. The docs are consistent that those were removed from the working design.

## What To Treat As Source Of Truth

Use the docs folder as the project brief, but distinguish between:

- implemented repo behavior
- planned experiments/paper claims
- reported results

Be strict about this distinction. Several docs mix these together.

For code-facing work, prefer these files:

- `docs/adr_fair_rlvr.md`
- `docs/tech_reward_function.md`
- `docs/tech_bbq_dataset.md`
- `docs/tech_grpo_training.md`
- `docs/paper_experiments.md`

If a claim in prose conflicts with code, call out the conflict instead of silently
copying the claim forward.

## Operational Ground Rules

### Data and splits

- BBQ is loaded from HuggingFace via `src/data.py`.
- The implemented split is a template-family-aware `90/10` split over the full base BBQ
  dataset, not category holdout training.
- Keep the training/eval `seed` aligned across all scripts. Seed mismatch causes split
  drift and can invalidate evaluation.
- Intersectional BBQ categories (`race_x_gender`, `race_x_ses`) are separate OOD eval,
  not part of the main train/eval split.

### Unknown / abstention handling

- The "Unknown" option is not always `(c)`.
- Use `unknown_label` from `src.data.get_unknown_label()`.
- Do not hardcode abstention as option index `2`.

### Reward function

- `R_fairness`: `+1` if predicted answer matches `answer_label`, else `0`
- `P_structural`: three rule-based penalties, `0.3` each
  - too-short or missing `<think>`
  - answer leak into `<think>`
  - content outside `<think>` / `<answer>`

### Training

- Main training entry point: `python -m src.train`
- Default intended config:
  - `lambda_fair=0.5`
  - `num_train_steps=3500`
  - `group_size=16`
  - `batch_size=8`
  - `gradient_accumulation=2`
  - `save_steps=500`
  - 4-bit `bitsandbytes` quantization during training
- LoRA target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Evaluation

- Main evaluation entry point: `python -m src.evaluate`
- Primary fairness metric for paper-quality evaluation is the official BBQ bias score:
  `2 * P(target_label | ambiguous) - 1`
- The simplified stereotype-error ratio is secondary and should not be presented as the
  canonical BBQ metric.
- Do not claim OOD generalization to held-out BBQ base categories. The docs support OOD
  claims only for WinoBias, StereoSet, and intersectional BBQ evaluation.

## Commands

```bash
# Install
pip install -e .
pip install -r requirements.txt

# Dry run
python -m src.train --dry-run

# Main training
python -m src.train --config configs/fair_rlvr.yaml

# Lambda sweep / overrides
python -m src.train --lambda-fair 0.1 --output-dir results/lambda_0.1

# Baselines
python -m src.baselines.baseline_model
python -m src.baselines.sft
python -m src.baselines.grpo_no_fairness

# Evaluation
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter --run-faithfulness

# Pipeline checks
python -m src.reward
python -m src.data
```

## Research-Claim Hygiene

When editing code, docs, or paper text:

- Do not present planned experiments as completed results.
- Do not present old 4-component reward equations as current.
- Do not claim DAPO features beyond what is clearly configured in this repo. The repo
  explicitly configures asymmetric clipping; stronger dynamic-sampling claims should be
  verified before repeating them.
- Do not claim "accepted IEEE paper" or fixed benchmark wins unless the user wants repo
  text to preserve that positioning. Treat that as project context, not a guaranteed fact.
- If citing numbers, prefer numbers that exist in repository artifacts under `results/`
  or in the active paper draft, and name the source.

## Important Repo Notes

- `src/train.py` builds a `ground_truth_map` keyed by formatted prompt strings. Changes to
  prompt formatting can break reward lookup if not updated carefully.
- `SYSTEM_PROMPT` in `src/data.py` is part of the training contract. Keep training and
  eval formatting aligned with it.
- `src/callbacks.py` contains some older descriptive comments. Treat executable behavior
  as authoritative over stale commentary.
- Existing `results/` directories contain prior outputs and should be treated as data,
  not as proof that every claim in docs is current.
