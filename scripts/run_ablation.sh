#!/bin/bash
# =============================================================================
# 一键跑所有消融实验（4 卡并行 + 自动评估）
#
# Usage:
#   bash scripts/run_ablation.sh                 # 默认 15 epoch
#   NUM_EPOCHS=5 bash scripts/run_ablation.sh    # 快速 5 epoch
#   GPU_COUNT=2 bash scripts/run_ablation.sh     # 2 卡并行
#   SKIP_EVAL=1 bash scripts/run_ablation.sh     # 只训练不评估
#
# 环境变量：
#   NUM_EPOCHS    训练轮数（默认 15）
#   GPU_COUNT     并行卡数（默认 4）
#   BASE_MODEL    基础模型路径（默认 Qwen2.5-Math-1.5B）
#   TRAIN_DATA    训练数据路径
#   TEST_DATA     测试数据路径
#   COT_ACC       Teacher CoT accuracy（用于 retention 计算）
#   COT_TOKENS    Teacher 平均输出 token 数
#   SKIP_EVAL     设为 1 跳过自动评估阶段
# =============================================================================

set -e

# -------- 配置（支持环境变量覆盖） --------
NUM_EPOCHS=${NUM_EPOCHS:-15}
GPU_COUNT=${GPU_COUNT:-4}
BASE_MODEL=${BASE_MODEL:-models/qwen2.5-math-1.5b}
TRAIN_DATA=${TRAIN_DATA:-data/gsm8k_train.json}
TEST_DATA=${TEST_DATA:-data/gsm8k_test.json}
COT_ACC=${COT_ACC:-0.83}
COT_TOKENS=${COT_TOKENS:-300}
SKIP_EVAL=${SKIP_EVAL:-0}

ABLATION_ROOT=checkpoints/ablation
RESULTS_ROOT=results/ablation

CONFIGS=(
    config/exp/ablation/full_loss.yaml
    config/exp/ablation/no_transition.yaml
    config/exp/ablation/no_anchor.yaml
    config/exp/ablation/no_bridge.yaml
    config/exp/ablation/transition_only.yaml
    config/exp/ablation/endpoint_only.yaml
    config/exp/ablation/k1.yaml
    config/exp/ablation/k2.yaml
    config/exp/ablation/k3.yaml
    config/exp/ablation/k5.yaml
    config/exp/ablation/layer_last1.yaml
    config/exp/ablation/layer_last4.yaml
    config/exp/ablation/layer_all.yaml
    config/exp/ablation/dist_cosine.yaml
    config/exp/ablation/dist_l2.yaml
)

mkdir -p "$ABLATION_ROOT" "$RESULTS_ROOT"

echo "========================================"
echo "  Ablation Run Configuration"
echo "========================================"
echo "  NUM_EPOCHS = $NUM_EPOCHS"
echo "  GPU_COUNT  = $GPU_COUNT"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  TRAIN_DATA = $TRAIN_DATA"
echo "  TEST_DATA  = $TEST_DATA"
echo "  Configs    = ${#CONFIGS[@]} experiments"
echo "========================================"

# -------- 阶段 1: 并行训练 --------
echo ""
echo "[Stage 1/2] Training ${#CONFIGS[@]} ablation experiments..."
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME=$(basename "$CFG" .yaml)
    GPU_ID=$((i % GPU_COUNT))
    SAVE_DIR="$ABLATION_ROOT/$NAME"

    echo "[GPU $GPU_ID] Training $NAME -> $SAVE_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train.py --config "$CFG" \
        model.name="$BASE_MODEL" \
        data.data_path="$TRAIN_DATA" \
        training.num_epochs="$NUM_EPOCHS" \
        training.batch_size=4 \
        training.gradient_accumulation_steps=4 \
        checkpoint.save_dir="$SAVE_DIR" \
        checkpoint.keep_top_k=999 \
        logging.use_wandb=false \
        > "$SAVE_DIR.train.log" 2>&1 &

    # 每 GPU_COUNT 个一批，等待当前批完成
    if (( (i + 1) % GPU_COUNT == 0 )); then
        wait
    fi
done
wait
echo "[Stage 1/2] All training jobs completed!"

# -------- 阶段 1.5: 自动选最优 checkpoint 并导出 final/ --------
echo ""
echo "[Stage 1.5] Selecting best val_loss checkpoint for each ablation..."
python scripts/select_best_checkpoint.py \
    --batch_root "$ABLATION_ROOT" \
    --base_model "$BASE_MODEL"

# -------- 阶段 2: 并行评估（可选） --------
if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL=1, skipping evaluation."
    exit 0
fi

echo ""
echo "[Stage 2/2] Evaluating all ablations with run_diagnostics.py..."
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME=$(basename "$CFG" .yaml)
    GPU_ID=$((i % GPU_COUNT))
    CKPT="$ABLATION_ROOT/$NAME/final"
    OUT="$RESULTS_ROOT/$NAME.json"

    if [ ! -d "$CKPT" ]; then
        echo "[SKIP] $NAME: final dir not found at $CKPT"
        continue
    fi

    echo "[GPU $GPU_ID] Evaluating $NAME -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_diagnostics.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --data_path "$TEST_DATA" \
        --no_chat_template \
        --cot_accuracy "$COT_ACC" \
        --cot_avg_tokens "$COT_TOKENS" \
        --show_samples 5 \
        --output "$OUT" \
        > "$RESULTS_ROOT/$NAME.eval.log" 2>&1 &

    if (( (i + 1) % GPU_COUNT == 0 )); then
        wait
    fi
done
wait

echo ""
echo "========================================"
echo "  All ablation experiments completed!"
echo "========================================"
echo "  Checkpoints: $ABLATION_ROOT/<name>/final"
echo "  Results:     $RESULTS_ROOT/<name>.json"
echo "  Train logs:  $ABLATION_ROOT/<name>.train.log"
echo "  Eval logs:   $RESULTS_ROOT/<name>.eval.log"
echo "========================================"
