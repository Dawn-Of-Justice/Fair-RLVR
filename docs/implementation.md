# Fair-RLVR Implementation Roadmap

## Phase 1: Local Setup
| Step | Task | Status |
|---|---|---|
| 1.1 | `src/data.py` — BBQ loading, 90/10 split, `unknown_label` field | ✅ Done |
| 1.2 | `src/reward.py` — Composite reward: `λ·R_fairness - P_structural` | ✅ Done |
| 1.3 | `src/evaluate.py` — BBQ accuracy, bias score, abstention rate (index-based, not string heuristic) | ✅ Done |
| 1.4 | `src/baselines/baseline_model.py` — Zero-shot Qwen2.5-3B on BBQ (renamed from zero_shot.py) | ✅ Done |
| 1.5 | `src/baselines/sft.py` — Supervised fine-tune on BBQ answers | ✅ Done |
| 1.6 | `src/baselines/grpo_no_fairness.py` — GRPO λ=0 ablation (replaces "correctness-only" baseline) | ✅ Done |
| 1.7 | Test `data.py` + `reward.py` + `evaluate.py` locally (no GPU needed) | ⬜ |

## Phase 2: Training Code
| Step | Task | Status |
|---|---|---|
| 2.1 | `src/train.py` — GRPO training loop with TRL GRPOTrainer + DAPO clipping | ✅ Done |
| 2.2 | `configs/` — YAML configs (fair_rlvr.yaml, lambda_sweep.yaml, dry_run.yaml) | ✅ Done |
| 2.3 | `src/callbacks.py` — Reward logging, training dynamics (6-phase), CoT sampling | ✅ Done |
| 2.4 | Bug fixes: CLI/config override logic, dtype, DAPO clip ratios, dynamics logger wiring | ✅ Done |
| 2.5 | Bug fix: abstention metric using index comparison via `unknown_label` field | ✅ Done |
| 2.6 | Bug fix: seed consistency across train/eval/baselines (data leakage prevention) | ✅ Done |
| 2.7 | Dry run with `--dry-run` flag (5 steps, verifies pipeline end-to-end) | ⬜ |

## Phase 3: GPU Training (H100 or A100 recommended for 3500 steps)
| Step | Task | Status | Est. Time |
|---|---|---|---|
| 3.1 | Push repo to GitHub, clone on cloud GPU | ⬜ | — |
| 3.2 | **Exp 1:** Run all 4 conditions (zero-shot, SFT, GRPO λ=0, Fair-RLVR λ=0.5) | ⬜ | ~4×14h |
| 3.3 | **Exp 2:** Lambda sweep (0.1, 0.3, 0.5, 0.7, 1.0) | ⬜ | ~5×14h |
| 3.4 | **Exp 3:** Training dynamics — checkpoints save automatically at 500-step intervals | ⬜ | (included in Exp 1) |
| 3.5 | **Exp 4:** Causal faithfulness test — `--run-faithfulness` flag in evaluate.py | ⬜ | ~1h post-train |
| 3.6 | **Exp 5:** OOD eval on WinoBias + StereoSet + intersectional BBQ | ⬜ | ~2h post-train |
| 3.7 | **Exp 6:** Bias amplification — bias score logged at each checkpoint step | ⬜ | (included in Exp 1) |
| 3.8 | MMLU + GSM8K alignment tax check | ⬜ | ~2h post-train |

## Phase 4: Paper Writing (after results)
| Step | Task | Status |
|---|---|---|
| 4.1 | Fill in results tables in `main.tex` | ⬜ |
| 4.2 | Generate training dynamics plots (reward curves, bias score over steps) | ⬜ |
| 4.3 | Cherry-pick CoT examples showing emergent de-biasing reasoning | ⬜ |
| 4.4 | Update paper equation: `R_total = λ·R_fairness - P_structural` (remove R_correctness, P_leak) | ⬜ |
| 4.5 | Write analysis + limitations with real numbers | ⬜ |
| 4.6 | Final paper review + submit | ⬜ |

## Key Architectural Decisions Made
| Decision | Choice | Rationale |
|---|---|---|
| Reward equation | `λ·R_fairness - P_structural` | R_correctness redundant post-format; P_leak not meaningful for single-letter MCQA |
| Training samples | 52,643 (full 90%) | More signal, proper generalization vs ~1K Med-RLVR style |
| GRPO group size | G=16 | Higher variance within group → more stable advantage normalization |
| Training steps | 3,500 | Enough for full reward dynamics curve; saves at 500-step intervals |
| λ=0 baseline | `grpo_no_fairness.py` | Directly tests whether fairness reward drives debiasing, not just GRPO structure |
| Abstention metric | Index comparison via `unknown_label` | String heuristic always returned 0% (model outputs "(a)" not "unknown") |
