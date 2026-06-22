#!/bin/bash
# =============================================================================
# 一键跑所有消融实验（4 卡并行）
# Usage: bash scripts/run_ablation.sh
# =============================================================================

set -e

CONFIGS=(
    config/exp/ablation/no_transition.yaml
    config/exp/ablation/no_anchor.yaml
    config/exp/ablation/no_bridge.yaml
    config/exp/ablation/transition_only.yaml
    config/exp/ablation/full_loss.yaml
    config/exp/ablation/k1.yaml
    config/exp/ablation/k2.yaml
    config/exp/ablation/k5.yaml
    config/exp/ablation/layer_last1.yaml
    config/exp/ablation/layer_last4.yaml
    config/exp/ablation/layer_all.yaml
    config/exp/ablation/dist_cosine.yaml
)

GPU_COUNT=4
for i in "${!CONFIGS[@]}"; do
    GPU_ID=$((i % GPU_COUNT))
    echo "Running ${CONFIGS[$i]} on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train.py --config "${CONFIGS[$i]}" \
        model.name=checkpoints/stage0_cot/final \
        data.data_path=data/gsm8k_train.json \
        logging.use_wandb=true &

    # 每 4 个一批，等待当前批完成
    if (( (i + 1) % GPU_COUNT == 0 )); then
        wait
    fi
done
wait
echo "All ablation experiments completed!"
