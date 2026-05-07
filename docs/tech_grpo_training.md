---
name: GRPO Training Setup and Pitfalls
description: GRPO algorithm details, hyperparameters, known failure modes, and DAPO fixes
type: project
---

# GRPO Training for Fair-RLVR

## Why GRPO (not PPO)
- No separate critic model needed → ~50% VRAM reduction
- Baseline = mean reward of a group of sampled outputs for same prompt
- Enables single T4 (16GB) training with 4-bit quantization

## Algorithm Summary
1. Sample G outputs per prompt (G=4 or 8)
2. Compute R_total for each output
3. Normalize: advantage_i = (R_i - mean(R)) / std(R)
4. Policy gradient update with clipping (PPO-style clip)
5. KL penalty term to prevent drift from base model

## Known Failure Modes

### Entropy Collapse (most dangerous)
- Model policy becomes too narrow → stops exploring
- All G outputs become nearly identical → no learning signal
- Source: DeepSeekMath paper
- Fix (DAPO): "Clip-Higher" — asymmetric clipping allows upward exploration
  ```
  clip_ratio_high = 0.28  (higher than standard 0.2)
  clip_ratio_low  = 0.20
  ```

### Reward Hacking Phases (from Med-RLVR)
Models go through phases during training:
1. Format Failure — can't produce valid tags
2. Verbose Formatter — writes "I am thinking..." repeatedly
3. Concise Structurer — writes real short logic
4. **The Hacker** — puts answer inside `<think>` block
5. **The Exploit** — finds formatting tricks for minimal-effort reward
6. Reintegrated Reasoning — stabilizes on real logic (target state)

### Response Length Collapse (from VMR-RLVR)
- Training on structured tasks collapses generation length
- Fix: mix in small % of standard RLHF data (~10%)

### Small Model Bias Amplification (from FairReason)
- 3B models: improved reasoning can *increase* bias manifestation
- Risk is real for Qwen2.5-3B
- Mitigation: monitor bias scores throughout training, not just at end

## Hyperparameters (as implemented in `configs/fair_rlvr.yaml` / `src/train.py`)
```python
model                 = "Qwen/Qwen2.5-3B-Instruct"
quantization          = "4-bit" (bitsandbytes nf4)  # NOT unsloth
lora_r                = 16
lora_alpha            = 32
lora_targets          = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
lr                    = 1e-5
group_size            = 16         # G=16 outputs per prompt (was 8 in early plan)
max_new_tokens        = 512
batch_size            = 8          # per-device (was 4 in early plan)
gradient_accumulation = 2          # effective batch = 16 (was 4 in early plan)
clip_ratio_high       = 0.28       # DAPO asymmetric clipping
clip_ratio_low        = 0.20
kl_coeff              = 0.01
training_steps        = 3500       # was 500-1000 in early plan
save_steps            = 500        # checkpoints at 500,1000,1500,2000,2500,3000,3500
train_ratio           = 0.9        # 90/10 split of full 58,492 BBQ dataset
seed                  = 42         # MUST match eval seed — see data leakage note
```

## Library Stack (actual)
- `trl` (HuggingFace) — GRPOTrainer
- `bitsandbytes` — 4-bit quantization (nf4). **NOT unsloth** — bitsandbytes used throughout
- `peft` — LoRA adapter
- `datasets` (HuggingFace) — BBQ loading from `Elfsong/BBQ`
- `sentence-transformers` — no longer used (P_leak removed from reward)
- `torch` — backend

## DAPO Techniques: Implemented vs Not
| Technique | Status | Notes |
|---|---|---|
| Clip-Higher (asymmetric ε) | ✅ Implemented | `epsilon=0.20`, `epsilon_high=0.28` in GRPOConfig |
| Token-level policy gradient | ✅ Implemented | TRL's GRPOTrainer uses token-level by default |
| Dynamic Sampling | ❌ Not implemented | Would filter all-correct / all-wrong groups; adds complexity |
| Entropy regularization bonus | ❌ Not implemented | Asymmetric clipping already handles entropy collapse |

## Data Leakage Warning
Training and evaluation both use `create_splits(train_ratio=0.9, seed=seed)`.
If the seeds differ between training and eval, the eval set may overlap with training data.
**Always run evaluation with `--seed 42` (or whatever seed was used in training).**
All baseline scripts now accept `--seed` and default to 42.

## Known Failure Mode: Small Model Bias Amplification
- FairReason showed 3B models can get *more* biased as reasoning improves at intermediate steps
- Track bias score every 100 steps via the `FairRLVRCallback` — not just at the end
- If amplification occurs, document the λ range and training step where it peaks
