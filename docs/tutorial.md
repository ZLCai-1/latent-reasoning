# Latent Reasoning 执行教程

## 环境准备

```bash
conda activate latent_reasoning
cd /Users/zlcai/latent_reasoning
```

---

## 完整执行流程

### Step 1: 验证 Pipeline 跑通（本地快速验证）

```bash
python scripts/test_e2e.py
```

**做了什么**：用随机初始化的小 GPT-2 跑完整流程（数据加载 → span 划分 → teacher 状态提取 → transition 训练 → loss 验证），不需要联网。

**预期输出**：
```
[1/5] Loading GPT-2 model... OK
[2/5] Preparing mini dataset... OK (7 samples)
[3/5] Teacher state extraction... OK
[4/5] Training loop:
  Epoch 1/3: loss=1.6989 (transition=0.2386, generation=5.1061)
  Epoch 2/3: loss=1.5506
  Epoch 3/3: loss=1.4718
[5/5] Loss trend: DECREASING ✓
ALL CHECKS PASSED!
```

**验证目标**：确认代码没有 bug，loss 能下降。

---

### Step 2: 验证 Loss 模块（单元测试）

```bash
python scripts/test_loss.py
```

**做了什么**：单独测试所有 loss 函数（transition_loss / anchor_loss / bridge_loss / generation_loss），验证梯度正常、无 NaN、边界情况处理正确。

**预期输出**：所有测试 PASS。

---

### 以下步骤在 4×3090 实例上执行（需联网下载模型和数据）

---

### Step 3: 数据预处理（生成带 spans 字段的训练数据）

```bash
python scripts/preprocess_data.py --input data/gsm8k_raw.json --output data/gsm8k_train.json --num_spans 3 --strategy fixed
```

**做了什么**：读取原始数据（question/answer/steps），将 steps 切分为 K 个 spans，输出新格式 JSON。

**输入格式**（原始数据）：
```json
{"question": "...", "answer": "...", "steps": ["step1", "step2", "step3"]}
```

**输出格式**（处理后）：
```json
{"question": "...", "answer": "...", "steps": ["step1", "step2", "step3"], "spans": [["step1", "step2"], ["step3"]]}
```

**参数说明**：
- `--num_spans`: K 值（将 CoT 切成几段）
- `--strategy`: 切分策略（`fixed` 均匀划分 / `random` 随机划分）

> 注意：训练时数据必须有 `spans` 字段，否则报错。

---

### Step 4: 训练 CoT Teacher（Stage 0）

```bash
python scripts/train.py --config config/exp/stage0_cot.yaml
```

**做了什么**：用显式 CoT 数据微调 GPT-2，得到一个能做数学推理的 teacher 模型。

**输出**：`checkpoints/stage0_cot/` 下的模型权重。

**配置文件解释** (`config/exp/stage0_cot.yaml`)：
- `model.name: "gpt2"` — 用 GPT-2 作为基础模型
- `data.span_strategy: "none"` — Stage 0 不切 span，用完整 CoT 训练
- `loss.generation_weight: 1.0` — 只有生成 loss，没有 transition loss

---

### Step 5: 提取 Teacher 隐状态

```bash
python scripts/extract_teacher_states.py \
    --config config/exp/stage1_transition.yaml \
    --teacher_path checkpoints/stage0_cot/final \
    --output_dir data/teacher_states
```

**做了什么**：用 Stage 0 训练好的 teacher 模型跑一遍所有训练数据，在每个 span boundary 位置提取 hidden state 并计算 ΔH，缓存到 HDF5 文件。

**输出**：`data/teacher_states/` 下的 `.h5` 文件（后续训练直接读取，不用重复计算）。

---

### Step 6: Transition Alignment 训练（Stage 1 — 核心实验）

```bash
python scripts/train.py --config config/exp/stage1_transition.yaml
```

**做了什么**：加载预缓存的 teacher states，训练 student 模型对齐状态转移量 ΔH。

**配置文件关键参数**：
- `model.num_latent_tokens: 3` — 用 3 个 latent token 替代完整 CoT
- `data.num_spans: 3` — CoT 被切成 3 段
- `loss.transition_weight: 0.7` — 状态转移 loss 权重
- `loss.generation_weight: 0.3` — 答案生成 loss 权重

**输出**：`checkpoints/stage1_transition/` 下的模型权重 + WandB 训练曲线。

**超参覆盖示例**（命令行直接改参数，不需要改配置文件）：
```bash
# 调整学习率和 batch_size
python scripts/train.py --config config/exp/stage1_transition.yaml \
    training.learning_rate=1e-4 \
    training.batch_size=8

# 换模型
python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name="Qwen/Qwen2.5-0.5B"

# 调 loss 权重
python scripts/train.py --config config/exp/stage1_transition.yaml \
    loss.transition_weight=0.5 \
    loss.generation_weight=0.5
```

---

### Step 7: 评估

```bash
python scripts/evaluate.py \
    --config config/exp/stage1_transition.yaml \
    --checkpoint checkpoints/stage1_transition/final \
    --split test
```

**做了什么**：加载训练好的模型，在测试集上生成答案并计算 accuracy / exact_match。

**输出**：打印准确率，保存结果到 JSON 文件。

---

## 超参搜索（9 组实验 — Plan Step 1.3）

在 4 张卡上并行跑：

```bash
# GPU 0: lr=1e-4
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/exp/stage1_transition.yaml \
    training.learning_rate=1e-4 loss.transition_weight=0.1 &

# GPU 1: lr=1e-4, weight=0.5
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/exp/stage1_transition.yaml \
    training.learning_rate=1e-4 loss.transition_weight=0.5 &

# GPU 2: lr=5e-5
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config config/exp/stage1_transition.yaml \
    training.learning_rate=5e-5 loss.transition_weight=0.1 &

# GPU 3: lr=5e-5, weight=0.5
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config config/exp/stage1_transition.yaml \
    training.learning_rate=5e-5 loss.transition_weight=0.5 &

wait
echo "所有实验完成"
```

---

## 文件对照表

| 你运行的命令 | 它内部调用了 | 作用 |
|------------|------------|------|
| `python scripts/test_e2e.py` | preprocessing → state_transition → loss_functions | 验证 pipeline |
| `python scripts/preprocess_data.py --input ... --output ...` | preprocessing.py | 生成带 spans 的训练数据 |
| `python scripts/train.py --config ...` | dataset → trainer → loss_functions | 训练模型 |
| `python scripts/extract_teacher_states.py` | base.py → state_extractor | 提取 teacher 状态 |
| `python scripts/evaluate.py` | base.py → evaluator → metrics | 评估准确率 |

---

## 配置文件一览

| 配置文件 | 用途 | 关键区别 |
|---------|------|---------|
| `config/base.yaml` | 默认参数模板 | 所有默认值 |
| `config/exp/stage0_cot.yaml` | Stage 0: CoT 预训练 | `span_strategy: "none"`, 只有 generation loss |
| `config/exp/stage1_transition.yaml` | Stage 1: Transition 对齐 | `span_strategy: "fixed"`, transition + generation loss |

---

## 总结：你需要记住的命令

```bash
# 本地验证（现在就能跑）
python scripts/test_e2e.py

# GPU 实例上的完整流程
python scripts/preprocess_data.py --input data/gsm8k_raw.json --output data/gsm8k_train.json --num_spans 3 --strategy fixed  # 1. 预处理数据
python scripts/train.py --config config/exp/stage0_cot.yaml           # 2. 训练 teacher
python scripts/extract_teacher_states.py --teacher_path checkpoints/stage0_cot/final  # 3. 提取状态
python scripts/train.py --config config/exp/stage1_transition.yaml    # 4. 训练 student
python scripts/evaluate.py --checkpoint checkpoints/stage1_transition/final           # 5. 评估
```
