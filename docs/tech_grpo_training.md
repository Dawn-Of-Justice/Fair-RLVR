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

## Recommended Hyperparameters (starting point)
```python
model         = "Qwen2.5-3B-Instruct"
quantization  = "4-bit" (bitsandbytes / unsloth)
lora_r        = 16
lora_alpha    = 32
lr            = 1e-5
group_size    = 8          # G outputs per prompt
max_new_tokens = 512       # for <think> block
batch_size    = 4
gradient_accumulation = 4  # effective batch = 16
clip_ratio_high = 0.28     # DAPO fix
clip_ratio_low  = 0.20
kl_coeff      = 0.01
training_steps = ~500-1000  # Med-RLVR used ~1K samples
```

## Library Stack
- `trl` (HuggingFace) — GRPOTrainer
- `unsloth` — 4-bit quantization + LoRA for T4
- `peft` — LoRA adapter
- `datasets` — BBQ loading
- `torch` — backend

## DAPO Techniques to Incorporate
1. **Clip-Higher**: asymmetric clip ratios (above)
2. **Dynamic Sampling**: filter out prompts where all G outputs get same reward (no gradient signal)
3. **Token-level policy gradient loss** instead of sequence-level (more stable)
4. **Entropy regularization**: add small entropy bonus to reward to prevent collapse
