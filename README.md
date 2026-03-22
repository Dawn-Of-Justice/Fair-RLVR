# Fair-RLVR: Teaching Reasoning Models to Be Fair via Verifiable Reward Signals

## What is Fair-RLVR?

Fair-RLVR applies Reinforcement Learning from Verifiable Rewards (RLVR) — the same technique behind DeepSeek-R1's emergent reasoning — to **fairness alignment**. Instead of relying on inconsistent human feedback (RLHF), we use the [BBQ benchmark](https://github.com/nyu-mll/BBQ) as an automated fairness verifier with ground-truth labels.

The model is trained to **reason through bias** in its chain-of-thought before answering, rather than memorizing "safe" responses.

## Key Idea

```
R_total = R_correctness + λ · R_fairness - β · R_hacking_penalty
```

| Component | What it does |
|---|---|
| `R_correctness` | Rewards valid `<think>` and `<answer>` format |
| `R_fairness` | Rewards matching BBQ ground-truth label |
| `R_hacking_penalty` | Penalizes fairness buzzwords without real logic |

## Why Not Just RLHF?

| Problem | RLHF | Fair-RLVR |
|---|---|---|
| Noisy labels | Human raters disagree | BBQ has objective ground truth |
| Over-refusal | Model refuses harmless questions | Model reasons through the question |
| Fake fairness | Model says what sounds safe | Model must reason correctly to get reward |
| Scalability | Expensive human annotation | Fully automated verifier |

## Project Structure

```
RLVR/
├── main.tex              # IEEE paper (LaTeX)
├── references.bib        # Bibliography
├── RLVR.png              # Methodology diagram
├── src/                  # Training code (TODO)
│   ├── train.py          # GRPO training loop
│   ├── reward.py         # Composite reward function
│   ├── data.py           # BBQ dataset loading
│   └── evaluate.py       # Evaluation pipeline
├── configs/              # Hyperparameter configs (TODO)
└── results/              # Experiment outputs (TODO)
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
- **Bias Score** — proportion of stereotype-consistent errors
- **Abstention Rate** — over-refusal detection
- **MMLU / GSM8K** — general reasoning (alignment tax check)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/Fair-RLVR.git
cd Fair-RLVR

# Install dependencies
pip install torch transformers trl peft unsloth datasets

# Train (TODO — code not yet implemented)
python src/train.py --lambda_fair 0.5 --beta_hack 0.1

# Evaluate
python src/evaluate.py --checkpoint outputs/checkpoint-1000
```

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Emergent reasoning via RL
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [BBQ Benchmark](https://arxiv.org/abs/2110.08193) — Bias Benchmark for QA
- [Med-RLVR](https://arxiv.org/abs/2502.xxxxx) — Emergent medical reasoning via RLVR
- [FairReason](https://arxiv.org/abs/2507.23067) — Balancing reasoning and social bias
- [DAPO](https://arxiv.org/abs/2503.14476) — Stable GRPO training at scale

## License

MIT
