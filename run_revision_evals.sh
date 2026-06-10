#!/usr/bin/env bash
# Revision evals for IEEE reviewer response — INFERENCE ONLY (no training).
# Addresses: external-benchmark generalization (R1.1/R2.1), the missing GRPO
# ablation row (Exp 1), and held-out metrics with bootstrap CIs.
#
# Budget notes:
#  - Fair-RLVR and SFT already have predictions.json -> reuse via --skip-inference;
#    they only pay the OOD inference pass, not the BBQ pass.
#  - Intersectional BBQ is 27,120 items; --ood-n-samples 2000 caps every OOD
#    benchmark at 2,000 (WinoBias/StereoSet are ~1.5-2k total anyway).
#  - bootstrap_ci is written into every metrics.json automatically (CPU, free).
set -euo pipefail

OOD_N=2000
# Inference batch size. 128 is comfortable for a merged bf16 3B model on an
# 80GB H100; drop to 64 if you hit OOM, or raise to 192 on a fresh GPU.
BS=128

echo "=== [1/3] Fair-RLVR lambda=0.5 : OOD only (reuse BBQ predictions) ==="
python -m src.evaluate \
  --checkpoint results/fair_rlvr/final_adapter \
  --output-dir results/eval_lambda_0.5 \
  --batch-size "$BS" \
  --skip-inference --run-ood --ood-n-samples "$OOD_N"

echo "=== [2/3] SFT baseline : OOD only (reuse BBQ predictions) ==="
python -m src.evaluate \
  --checkpoint results/sft/adapter \
  --output-dir results/sft \
  --batch-size "$BS" \
  --skip-inference --run-ood --ood-n-samples "$OOD_N"

echo "=== [3/3] GRPO ablation (grpo_correctness_only) : full BBQ eval + OOD ==="
python -m src.evaluate \
  --checkpoint results/grpo_correctness_only/final_adapter \
  --output-dir results/eval_grpo_correctness_only \
  --batch-size "$BS" \
  --run-ood --ood-n-samples "$OOD_N"

echo "=== DONE ==="
echo "Check each metrics.json for:"
echo "  - ood.winobias.winobias_bias_score (should be a number, not an error)"
echo "  - ood.stereoset.label_order  (confirms StereoSet label mapping)"
echo "  - ood.intersectional_bbq.accuracy_disambiguated (should be plausible, not ~0)"
echo "  - bootstrap_ci.consistency_check (should say OK)"
