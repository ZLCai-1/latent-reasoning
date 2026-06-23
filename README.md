# Latent Reasoning: State Transition Alignment

面向高效大模型推理的状态转移对齐式 Latent Reasoning Token 学习

## 核心思想

用少量 latent tokens 替代显式 Chain-of-Thought 推理链，通过对齐 Teacher 模型内部隐状态的转移量（ΔH）来训练 Student 模型。

## 项目结构

```
src/
├── models/          # 模型封装 + 状态转移模块 + 损失函数
├── data/            # 数据加载 + 预处理 + Teacher状态提取
├── training/        # 训练循环 + 课程学习
└── eval/            # 评估

scripts/
├── download_gsm8k.py          # 下载 GSM8K 数据集
├── preprocess_data.py         # 数据预处理（生成 spans）
├── train.py                   # 训练入口
├── extract_teacher_states.py  # 提取 Teacher 隐状态
├── evaluate.py                # 评估
├── run_ablation.sh            # 一键跑消融实验
├── run_baselines.py           # 对比基线
├── test_e2e.py                # 端到端验证
└── test_loss.py               # Loss 单元测试

config/
├── base.yaml                  # 默认配置
└── exp/
    ├── stage0_cot.yaml        # Stage 0: CoT Teacher 训练
    ├── stage1_transition.yaml # Stage 1: Transition Alignment
    ├── ablation/              # 14 个消融实验配置
    └── baselines/             # 3 个对比基线配置
```

## 快速开始

### 环境安装（4x3090 服务器）

```bash
git clone https://github.com/ZLCai-1/latent-reasoning.git
cd latent-reasoning
conda create -n latent_reasoning python=3.10 -y
conda activate latent_reasoning
pip install -r requirements.txt
```

### 完整训练流程

> 服务器访问 HuggingFace 受限时，所有下载命令需加前缀 `HF_ENDPOINT=https://hf-mirror.com`。
> 也可在 shell 中一次性设置：`export HF_ENDPOINT=https://hf-mirror.com`

```bash
# (可选) 全局设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 1. 下载 GSM8K 数据
python scripts/download_gsm8k.py --output data/gsm8k_train.json --split train
python scripts/download_gsm8k.py --output data/gsm8k_test.json --split test

# 2. 数据预处理（就地生成 spans 字段）
python scripts/preprocess_data.py --input data/gsm8k_train.json --output data/gsm8k_train.json --num_spans 3 --strategy fixed
python scripts/preprocess_data.py --input data/gsm8k_test.json --output data/gsm8k_test.json --num_spans 3 --strategy fixed

# 3. 准备 Teacher 模型（二选一）
#
# 方案 A：直接用标准 GPT-2 训 Stage 0（推荐，简单可靠）
# 确定gpt2的路径
ls models/gpt2/models--gpt2/snapshots/

python scripts/train.py --config config/exp/stage0_cot.yaml \
    model.name=models/gpt2/models--gpt2/snapshots/（上一步得到的哈希值）  \
    data.data_path=data/gsm8k_train.json

# 4. 提取 Teacher 隐状态（全量数据）
python scripts/extract_teacher_states.py \
    --config config/exp/stage1_transition.yaml \
    --teacher_path models/codi-gpt2 \
    --output_dir data/teacher_states

# 5. Stage 1: Transition Alignment 训练（全量，15 epochs curriculum）
python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/codi-gpt2 \
    data.data_path=data/gsm8k_train.json

# 6. 评估
python scripts/evaluate.py \
    --config config/exp/stage1_transition.yaml \
    --checkpoint checkpoints/stage1_transition/final \
    --split test \
    --data_path data/gsm8k_test.json
```

### 超参搜索（4 卡并行）

```bash
# 9 组实验：lr x transition_weight
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/codi-gpt2 data.data_path=data/gsm8k_train.json \
    training.learning_rate=1e-4 loss.transition_weight=0.1 \
    checkpoint.save_dir=checkpoints/search/lr1e4_tw01 &

CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/codi-gpt2 data.data_path=data/gsm8k_train.json \
    training.learning_rate=1e-4 loss.transition_weight=0.5 \
    checkpoint.save_dir=checkpoints/search/lr1e4_tw05 &

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/codi-gpt2 data.data_path=data/gsm8k_train.json \
    training.learning_rate=5e-5 loss.transition_weight=0.1 \
    checkpoint.save_dir=checkpoints/search/lr5e5_tw01 &

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/codi-gpt2 data.data_path=data/gsm8k_train.json \
    training.learning_rate=5e-5 loss.transition_weight=0.5 \
    checkpoint.save_dir=checkpoints/search/lr5e5_tw05 &

wait
```

### 消融实验

```bash
bash scripts/run_ablation.sh
```

### 对比基线

```bash
python scripts/run_baselines.py --data_path data/gsm8k_train.json --model_name models/codi-gpt2
```

## 训练数据格式

```json
{
  "question": "Tom has 5 apples...",
  "answer": "3",
  "steps": ["Tom starts with 5.", "He gives away 2.", "5 - 2 = 3."],
  "spans": [["Tom starts with 5.", "He gives away 2."], ["5 - 2 = 3."]]
}
```

## 4 个核心 Loss

| Loss | 作用 | 权重 |
|------|------|------|
| transition_loss | 对齐状态转移量 ΔH | 0.5-0.8 |
| anchor_loss | 防止 hidden state 漂移 | 0.05-0.2 |
| bridge_loss | 缓解 exposure mismatch（3项公式） | 0.05-0.1 |
| generation_loss | 答案生成交叉熵 | 0.2-0.3 |

## 硬件需求

- 4× RTX 3090 24GB
- CUDA 12.2+
- Python 3.10+

## 技术栈

- PyTorch 2.0+
- HuggingFace Transformers
- OmegaConf (配置管理)
- WandB (实验追踪)
- DeepSpeed (多卡训练)
