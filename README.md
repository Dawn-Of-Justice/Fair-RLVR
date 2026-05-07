# Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals

## What is Fair-RLVR?

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) — the same technique behind DeepSeek-R1's emergent reasoning — to **fairness alignment**. Instead of relying on inconsistent human feedback (RLHF), we use the [BBQ benchmark](https://github.com/nyu-mll/BBQ) as an automated fairness verifier with ground-truth labels.

The model is trained to **reason through bias** in its chain-of-thought before answering, rather than memorizing "safe" responses.

## Key Idea

```
R_total = λ · R_fairness - P_structural
```

| Component | What it does |
|---|---|
| `R_fairness` | Rewards matching BBQ ground-truth label (+1.0 / 0) |
| `P_structural` | Penalizes format violations: answer leaked in `<think>`, reasoning < 20 tokens, content outside tags, missing `<answer>` tag (0.3 each, max 1.2) |
| `λ` | Fairness signal weight — default 0.5, ablated across {0.1, 0.3, 0.5, 0.7, 1.0} |

Reward range: `[−1.2, 0.5]` at λ=0.5. `R_correctness` and `P_leak` (Sentence-BERT) were removed from an earlier 4-component design — both were found to be redundant for single-letter MCQA (see [docs/adr_fair_rlvr.md](docs/adr_fair_rlvr.md)).

## Why Not Just RLHF?

| Problem | RLHF | Fair-RLVR |
|---|---|---|
| Noisy labels | Human raters disagree; annotator bias is amplified by RL | BBQ has objective ground truth — no human in the reward loop |
| Over-refusal | Model refuses harmless questions | Model reasons through the question to earn reward |
| Fake fairness | Model says what sounds safe | Model must answer correctly on bias-sensitive questions |
| Scalability | Expensive human annotation | Fully automated verifier — 58K labeled examples |

## Results

> **Status:** Code complete and verified. GPU training pending. Results will be filled in after Experiments 1–6 complete.

### Main Experiment (Exp 1)

| Condition | BBQ-Ambig | BBQ-Disambig | Bias Score (official) | MMLU | GSM8K |
|---|---|---|---|---|---|
| Zero-shot baseline | | | | | |
| SFT baseline | | | | | |
| GRPO λ=0 (no fairness reward) | | | | | |
| **Fair-RLVR λ=0.5 (ours)** | | | | | |

### Lambda Ablation (Exp 2)

| λ | BBQ-Ambig | BBQ-Disambig | Bias Score (official) | Abstention Rate |
|---|---|---|---|---|
| 0.1 | | | | |
| 0.3 | | | | |
| 0.5 | | | | |
| 0.7 | | | | |
| 1.0 | | | | |

## Project Structure

```
Fair-RLVR/
├── main.tex                # IEEE paper (LaTeX)
├── references.bib          # Bibliography
├── RLVR.png                # Methodology diagram
├── setup.py                # Package install
├── requirements.txt        # Dependencies
├── src/
│   ├── train.py            # GRPO training loop
│   ├── reward.py           # Composite reward function
│   ├── data.py             # BBQ dataset loading & template-family splits
│   ├── evaluate.py         # Evaluation pipeline
│   ├── callbacks.py        # Training callbacks & logging
│   └── baselines/
│       ├── baseline_model.py   # Zero-shot baseline (no training)
│       ├── sft.py              # SFT baseline
│       └── grpo_no_fairness.py # GRPO λ=0 ablation (R_total = −P_structural)
├── configs/
│   ├── fair_rlvr.yaml      # Main experiment config (3,500 steps)
│   ├── lambda_sweep.yaml   # λ ablation configs
│   └── dry_run.yaml        # Quick 5-step pipeline test
└── results/                # Experiment outputs (metrics only, weights excluded)
    ├── baseline_model/
    ├── sft/
    ├── grpo_no_fairness/
    └── fair_rlvr/
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/Dawn-Of-Justice/RLVR.git
cd RLVR
pip install -e .
pip install -r requirements.txt

# flash-attn must be built against your installed CUDA/torch versions.
# If the above fails for flash-attn, install it separately:
pip install flash-attn --no-build-isolation

# Step 1: Verify pipeline (5 steps — still loads the full model, GPU recommended)
python -m src.train --dry-run

# Step 2: Verify data split (should show ~52,643 train / ~5,849 eval)
python -m src.data

# Step 3: Verify reward function
python -m src.reward

# Step 4: Run baselines (use --seed 42 on all to ensure same train/eval split)
python -m src.baselines.baseline_model --seed 42
python -m src.baselines.sft --seed 42
python -m src.baselines.grpo_no_fairness --seed 42

# Step 5: Main training (Fair-RLVR λ=0.5, 3,500 steps)
python -m src.train --config configs/fair_rlvr.yaml

# Step 6: Lambda sweep
python -m src.train --lambda-fair 0.1 --output-dir results/lambda_0.1
python -m src.train --lambda-fair 0.3 --output-dir results/lambda_0.3
python -m src.train --lambda-fair 0.7 --output-dir results/lambda_0.7
python -m src.train --lambda-fair 1.0 --output-dir results/lambda_1.0

# Step 7: Evaluate (full eval set, ~5,849 samples)
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter

# Step 8: Interventional sensitivity test (Experiment 4)
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter --run-faithfulness
```

## Experiments

| # | Experiment | Question |
|---|---|---|
| 1 | Main Result | Does Fair-RLVR reduce bias without degrading reasoning? |
| 2 | Lambda Ablation | What λ balances fairness vs utility? |
| 3 | Training Dynamics | Do Med-RLVR's 6 phases appear in fairness training? |
| 4 | Interventional Sensitivity | Does the model's answer depend on CoT content, or is it internalized? |
| 5 | OOD Generalization | Does fairness transfer to WinoBias / StereoSet / intersectional BBQ? |
| 6 | Bias Amplification Check | Does improved reasoning increase bias at intermediate training steps? |

## Evaluation Metrics

- **BBQ Bias Score (official, primary)** — `2 × P(model picks target_label | ambiguous) − 1`. Range [−1, 1]. 0 = unbiased; positive = stereotype-biased.
- **BBQ Accuracy (Ambiguous)** — % correct on ambiguous context (correct answer is always "Unknown")
- **BBQ Accuracy (Disambiguated)** — % correct when evidence is provided
- **Abstention Rate** — measured via per-question `unknown_label` index (not string heuristic — the "Unknown" option is not always option (c))
- **Interventional Sensitivity Score** — P(correct | real CoT) − P(correct | corrupted CoT)

## Training Setup

| Component | Specification |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit bitsandbytes nf4 (training); float16 (eval) |
| Adaptation | LoRA (r=16, α=32, 7 target modules) |
| Algorithm | GRPO G=16 with DAPO asymmetric clipping (ε_low=0.20, ε_high=0.28) |
| Dataset | Full BBQ — 52,643 training / 5,849 eval (template-family-aware 90/10 split, seed=42) |
| Training Steps | 3,500 (checkpoint every 500 steps) |
| Hardware | NVIDIA H100 80GB (recommended); A100 40GB feasible |

## Paper

```bibtex
@inproceedings{fairrlvr2026,
  title={Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals},
  author={Salo, E S and Ravi, Arjun G and Devanand, A and Rini Susan, V S},
  booktitle={IEEE Conference},
  year={2026}
}
```

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Emergent reasoning via RL
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [BBQ Benchmark](https://arxiv.org/abs/2110.08193) — Bias Benchmark for QA
- [FairReason](https://arxiv.org/abs/2507.23067) — Balancing reasoning and social bias in MLLMs
- [DAPO](https://arxiv.org/abs/2503.14476) — Stable GRPO training at scale
- [Med-RLVR](https://arxiv.org/abs/2502.19655) — RLVR for medical reasoning; source of 6-phase training taxonomy
- [Tarek et al.](https://arxiv.org/abs/2509.15557) — Reward hacking mitigation via composite rewards

## License

This project is licensed under the [MIT License](LICENSE).
