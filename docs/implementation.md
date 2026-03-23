# Fair-RLVR Implementation Roadmap

## Phase 1: Local Setup
| Step | Task | Status |
|---|---|---|
| 1.1 | `src/data.py` — BBQ loading + splits | ✅ Done |
| 1.2 | `src/reward.py` — Composite reward function | ✅ Done |
| 1.3 | `src/evaluate.py` — Evaluation pipeline (BBQ accuracy, bias score, abstention rate) | ✅ Done |
| 1.4 | `src/baselines/zero_shot.py` — Run Qwen2.5-3B on BBQ with no training | ✅ Done |
| 1.5 | `src/baselines/sft.py` — Supervised fine-tune on BBQ answers | ✅ Done |
| 1.6 | Test `data.py` + `reward.py` + `evaluate.py` locally (no GPU needed) | ⬜ |

## Phase 2: Training Code (Local, no GPU needed to write)
| Step | Task | Status |
|---|---|---|
| 2.1 | `src/train.py` — GRPO training loop with TRL's GRPOTrainer | ⬜ |
| 2.2 | `configs/` — YAML configs for each experiment (lambda values, baselines) | ⬜ |
| 2.3 | `src/callbacks.py` — Logging: reward per step, entropy, abstention rate, CoT samples at checkpoints | ⬜ |
| 2.4 | Dry run with 5 samples to verify the pipeline doesn't crash | ⬜ |

## Phase 3: Move to Lightning.ai (GPU needed)
| Step | Task | Status |
|---|---|---|
| 3.1 | Push repo to GitHub, clone on Lightning.ai | ⬜ |
| 3.2 | **Experiment 1:** Run all 4 conditions (zero-shot, SFT, GRPO-only, Fair-RLVR) → ~16h | ⬜ |
| 3.3 | **Experiment 2:** Lambda sweep (0.1, 0.3, 0.5, 0.7, 1.0) → ~20h | ⬜ |
| 3.4 | **Experiment 3:** Save checkpoints at 100/250/500/1000 steps, log training dynamics | ⬜ |
| 3.5 | **Experiment 4:** Causal faithfulness test (permute CoT, measure answer degradation) | ⬜ |
| 3.6 | **Experiment 5:** OOD eval on WinoBias + StereoSet + held-out BBQ | ⬜ |
| 3.7 | **Experiment 6:** Bias amplification monitoring | ⬜ |

## Phase 4: Paper Writing (after results)
| Step | Task | Status |
|---|---|---|
| 4.1 | Fill in results tables in `main.tex` | ⬜ |
| 4.2 | Generate training dynamics plots (reward curves, entropy, abstention) | ⬜ |
| 4.3 | Cherry-pick CoT examples showing emergent de-biasing | ⬜ |
| 4.4 | Write analysis + limitations with real numbers | ⬜ |
| 4.5 | Final paper review + submit | ⬜ |
