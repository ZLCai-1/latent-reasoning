# Latent Reasoning: State-Transition Aligned Latent Reasoning Token Learning

用少量 latent tokens 替代显式 CoT 推理链，通过对齐 teacher 模型内部状态的"转移量"（ΔH）进行学习。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Stage 0 — 训练 CoT Teacher

```bash
python scripts/train.py --config config/exp/stage0_cot.yaml
```

### 3. 提取 Teacher Hidden States

```bash
python scripts/extract_teacher_states.py \
    --config config/exp/stage1_transition.yaml \
    --teacher_path checkpoints/stage0_cot/final \
    --output_dir teacher_states
```

### 4. Stage 1 — Transition Alignment 训练

```bash
python scripts/train.py --config config/exp/stage1_transition.yaml
```

### 5. 评估

```bash
python scripts/evaluate.py \
    --config config/exp/stage1_transition.yaml \
    --checkpoint checkpoints/stage1_transition/final \
    --split test
```

## 项目结构

```
├── config/             # YAML 配置文件
├── src/
│   ├── models/         # 模型 wrapper + 状态转移 + 损失函数
│   ├── data/           # 数据加载 + 预处理 + 状态抽取
│   ├── training/       # 训练循环 + 课程学习
│   └── eval/           # 评估器 + 指标
├── scripts/            # CLI 入口脚本
└── requirements.txt
```

## 硬件需求

- 4× RTX 3090 24GB (推荐)
- CUDA 12.2+
- Python 3.10+
