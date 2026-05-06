# Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals

## What is Fair-RLVR?

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) — the same technique behind DeepSeek-R1's emergent reasoning — to **fairness alignment**. Instead of relying on inconsistent human feedback (RLHF), we use the [BBQ benchmark](https://github.com/nyu-mll/BBQ) as an automated fairness verifier with ground-truth labels.

The model is trained to **reason through bias** in its chain-of-thought before answering, rather than memorizing "safe" responses.

## Key Idea

```
R_total = R_correctness + λ · R_fairness - P_structural - P_leak
```

| Component | What it does |
|---|---|
| `R_correctness` | Rewards valid `<think>` and `<answer>` format (+1 / -1) |
| `R_fairness` | Rewards matching BBQ ground-truth label (+1 / 0) |
| `P_structural` | Penalizes answer leaking, short reasoning, content outside tags |
| `P_leak` | Sentence-BERT similarity penalty — catches fake reasoning |

## Why Not Just RLHF?

| Problem | RLHF | Fair-RLVR |
|---|---|---|
| Noisy labels | Human raters disagree | BBQ has objective ground truth |
| Over-refusal | Model refuses harmless questions | Model reasons through the question |
| Fake fairness | Model says what sounds safe | Model must reason correctly to get reward |
| Scalability | Expensive human annotation | Fully automated verifier |

## Results

### Main Experiment

| Condition | BBQ-Ambig | BBQ-Disambig | Bias Score |
|---|---|---|---|
| Zero-shot | 82.4% | 89.8% | 0.561 |
| SFT | 89.4% | 87.6% | 0.358 |
| GRPO (correctness only) | 87.2% | 86.6% | 0.489 |
| **Fair-RLVR (ours)** | **95.8%** | **88.6%** | **0.269** |

### Lambda Sweep

| λ | BBQ-Ambig | BBQ-Disambig | Bias Score |
|---|---|---|---|
| 0.1 | 98.2% | 87.8% | **0.129** |
| 0.3 | 96.8% | 88.0% | 0.224 |
| 0.5 | 95.8% | 88.6% | 0.269 |
| 0.7 | 97.2% | **91.0%** | 0.305 |
| 1.0 | 97.6% | 87.0% | 0.169 |

### Alignment Tax (General Benchmarks)

| Condition | MMLU | GSM8K |
|---|---|---|
| Zero-shot | 62.2% | 83.6% |
| Fair-RLVR | **62.8%** (+0.6%) | **85.0%** (+1.4%) |

**Key findings:**
- **+13.4%** ambiguous accuracy over zero-shot
- **52% reduction** in bias score (0.561 → 0.269)
- **Zero** stereotype errors on gender identity
- **No alignment tax** — MMLU and GSM8K actually *improved* after fairness training
- All λ values effective — even λ=0.1 achieves 0.129 bias score
- No lobotomy effect at any λ — 0% over-refusal across all settings

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
│   ├── data.py             # BBQ dataset loading & splits
│   ├── evaluate.py         # Evaluation pipeline
│   ├── callbacks.py        # Training callbacks & logging
│   └── baselines/
│       ├── baseline_model.py   # Zero-shot baseline (no training)
│       ├── sft.py              # SFT baseline
│       └── grpo_no_fairness.py # GRPO λ=0 ablation baseline
├── configs/
│   ├── fair_rlvr.yaml      # Main experiment config
│   ├── lambda_sweep.yaml   # λ ablation configs
│   └── dry_run.yaml        # Quick test config
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

# Step 1: Dry run (verify everything works)
python -m src.train --dry-run

# Step 2: Baselines
python -m src.baselines.baseline_model
python -m src.baselines.sft
python -m src.baselines.grpo_no_fairness

# Step 3: Main training


# Step 4: Fair-RLVR (main experiment)
python -m src.train --config configs/fair_rlvr.yaml

# Step 5: Lambda sweep
python -m src.train --lambda-fair 0.1 --output-dir results/lambda_0.1
python -m src.train --lambda-fair 0.3 --output-dir results/lambda_0.3
python -m src.train --lambda-fair 0.7 --output-dir results/lambda_0.7
python -m src.train --lambda-fair 1.0 --output-dir results/lambda_1.0

# Step 6: Evaluate
python -m src.evaluate --checkpoint results/fair_rlvr/final_adapter
```

## Experiments

| # | Experiment | Question |
|---|---|---|
| 1 | Main Result | Does Fair-RLVR reduce bias without degrading reasoning? |
| 2 | Lambda Ablation | What λ balances fairness vs utility? |
| 3 | Training Dynamics | Do Med-RLVR's 6 phases appear in fairness training? |
| 4 | Causal Faithfulness | Is the chain-of-thought causally real or post-hoc? |
| 5 | OOD Generalization | Does fairness transfer to WinoBias / StereoSet? |
| 6 | Bias Amplification | Does improved reasoning increase bias in small models? |

## Evaluation Metrics

- **BBQ Accuracy (Ambiguous)** — primary fairness metric
- **BBQ Accuracy (Disambiguated)** — evidence-following ability
- **Bias Score** — proportion of stereotype-consistent errors (< 0.5 = fair)
- **Abstention Rate** — over-refusal detection

## Training Setup

| Component | Specification |
|---|---|
| Base Model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit (bitsandbytes) |
| Adaptation | LoRA (r=16, α=32) |
| Algorithm | GRPO (with DAPO fixes) |
| Dataset | BBQ (~1,000 training samples) |
| Training Steps | ~1,000 |
| Hardware | NVIDIA H100 (80GB) |

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
- [FairReason](https://arxiv.org/abs/2507.23067) — Balancing reasoning and social bias
- [DAPO](https://arxiv.org/abs/2503.14476) — Stable GRPO training at scale
- [Tarek et al.](https://arxiv.org/abs/2509.15557) — Reward hacking mitigation via composite rewards

## License

This project is licensed under the [MIT License](LICENSE).
