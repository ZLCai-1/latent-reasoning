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
├── test_e2e.py                # 端到端验证
└── test_loss.py               # Loss 单元测试

config/
├── base.yaml                  # 默认配置
└── exp/
    ├── stage0_cot.yaml        # Stage 0: CoT Teacher 训练
    └── stage1_transition.yaml # Stage 1: Transition Alignment
```

## 快速开始

### 环境安装

```bash
conda create -n latent_reasoning python=3.10 -y
conda activate latent_reasoning
pip install -r requirements.txt
```

### 完整流程

```bash
# 1. 下载数据
HF_ENDPOINT=https://hf-mirror.com python scripts/download_gsm8k.py --output data/gsm8k_raw.json --split train

# 2. 数据预处理
python scripts/preprocess_data.py --input data/gsm8k_raw.json --output data/gsm8k_train.json --num_spans 3 --strategy fixed

# 3. 训练 CoT Teacher (Stage 0)
python scripts/train.py --config config/exp/stage0_cot.yaml model.name=gpt2 data.data_path=data/gsm8k_train.json logging.use_wandb=false

# 4. 提取 Teacher 隐状态
python scripts/extract_teacher_states.py --config config/exp/stage1_transition.yaml --teacher_path checkpoints/stage0_cot/final --output_dir data/teacher_states

# 5. Transition Alignment 训练 (Stage 1)
python scripts/train.py --config config/exp/stage1_transition.yaml model.name=checkpoints/stage0_cot/final data.data_path=data/gsm8k_train.json logging.use_wandb=false

# 6. 评估
python scripts/evaluate.py --config config/exp/stage1_transition.yaml --checkpoint checkpoints/stage1_transition/final --split test
```

### 本地快速验证（不需要 GPU）

```bash
python scripts/test_e2e.py
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

## 技术栈

- PyTorch 2.0+
- HuggingFace Transformers
- OmegaConf (配置管理)
- WandB (实验追踪)
