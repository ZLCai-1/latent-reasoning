# Latent Reasoning: State Transition Alignment

面向高效大模型推理的状态转移对齐式 Latent Reasoning Token 学习

## 核心思想

用少量 latent tokens 替代显式 Chain-of-Thought 推理链，通过对齐 Teacher 模型内部隐状态的转移量（ΔH）来训练 Student 模型。

> **注意**：latent token 的作用是节省**推理阶段（prefill）**的 token 占用，而非减少输出 token 长度。

## 项目结构

```
src/
├── models/          # 模型封装 + 状态转移模块 + 损失函数
├── data/            # 数据加载 + 预处理 + Teacher状态提取
├── training/        # 训练循环 + 课程学习
└── eval/            # 评估器（evaluator.py） + 诊断指标（diagnostics.py）

scripts/
├── download_gsm8k.py          # 下载 GSM8K 数据集
├── preprocess_data.py         # 数据预处理（生成 spans）
├── train.py                   # 训练入口
├── extract_teacher_states.py  # 提取 Teacher 隐状态
├── evaluate.py                # 快速评估（accuracy + 效率）
├── run_diagnostics.py         # 完整诊断（§5.6 全部指标 + 定性样本）
├── verify_pipeline.py         # 评估管线最小验证
├── run_ablation.sh            # 一键跑消融实验
└── run_baselines.py           # 对比基线

config/
├── base.yaml                  # 默认配置
└── exp/
    ├── stage0_cot.yaml        # Stage 0: CoT Teacher 训练
    ├── stage1_transition.yaml # Stage 1: Transition Alignment
    ├── ablation/              # 14 个消融实验配置
    └── baselines/             # 3 个对比基线配置
```

## 环境安装

```bash
git clone https://github.com/ZLCai-1/latent-reasoning.git
cd latent-reasoning
conda create -n latent_reasoning python=3.10 -y
conda activate latent_reasoning
pip install -r requirements.txt
```

> 服务器访问 HuggingFace 受限时需设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 完整流程

### 1. 下载 GSM8K 数据

```bash
python scripts/download_gsm8k.py --output data/gsm8k_train.json --split train
python scripts/download_gsm8k.py --output data/gsm8k_test.json --split test
```

**参数说明**：
- `--output`：输出 JSON 文件路径
- `--split`：`train` 或 `test`

---

### 2. 数据预处理（生成 spans 字段）

```bash
python scripts/preprocess_data.py --input data/gsm8k_train.json --output data/gsm8k_train.json --num_spans 3 --strategy fixed
python scripts/preprocess_data.py --input data/gsm8k_test.json --output data/gsm8k_test.json --num_spans 3 --strategy fixed
```

**参数说明**：
- `--input`：输入文件路径
- `--output`：输出文件路径（可与 input 相同，就地修改）
- `--num_spans`：每条样本切分的 span 数 K（默认 3）
- `--strategy`：切分策略，`fixed` / `random` / `none`

---

### 3. 准备 Teacher 模型

**方案 A（推荐）：Qwen2.5-Math-1.5B-Instruct（GSM8K accuracy ~83%）**

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-Math-1.5B-Instruct', local_dir='models/qwen2.5-math-1.5b', local_dir_use_symlinks=False)
"
```

**方案 B：自训练 Stage 0 CoT Teacher（GPT-2 轻量验证）**

```bash
python scripts/train.py --config config/exp/stage0_cot.yaml
```

**参数说明**（通过 OmegaConf 命令行覆盖）：
- `data.data_path=<path>` —— 训练数据路径
- `data.max_samples=<int>` —— 限制样本数（0=全部，调试用）
- `training.num_epochs=<int>` —— 训练轮数
- `training.batch_size=<int>` —— 批大小
- `model.name=<path>` —— 基础模型路径

---

### 4. 评估 Teacher 模型（获取 CoT baseline）

```bash
python scripts/evaluate.py \
    --config config/exp/stage0_cot.yaml \
    --checkpoint models/qwen2.5-math-1.5b \
    --data_path data/gsm8k_test.json
```

**参数说明**：
- `--config`：配置文件（必需）
- `--checkpoint`：模型路径（**必需**）
- `--data_path`：评估数据路径
- `--split`：`test` / `train`（默认 `test`）
- `--batch_size`：评估批大小（默认 32）
- `--max_new_tokens`：最大生成 token 数（默认512）
- `--max_samples`：评估样本上限（0=全量，调试用 10-20）
- `--show_samples`：终端打印定性样本数（默认10）
- `--cot_baseline_tokens`：CoT 输出 token 基线（默认200）
- `--output`：结果 JSON 保存路径

**记录输出中的 `Accuracy` 和 `Avg Output Tokens`**，后续诊断需要传入这两个数。

---

### 5. 提取 Teacher 隐状态（HDF5 缓存）

```bash
python scripts/extract_teacher_states.py \
    --config config/exp/stage1_transition.yaml \
    --teacher_path models/qwen2.5-math-1.5b \
    --output_dir data/teacher_states
```

**参数说明**：
- `--config`：训练配置（提供 layer_ids 等）
- `--teacher_path`：Teacher 模型路径
- `--output_dir`：HDF5 输出目录
- `--batch_size`：提取批大小（默认从 config 读取）

---

### 6. Stage 1: Transition Alignment 训练

```bash
python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/qwen2.5-math-1.5b \
    data.data_path=data/gsm8k_train.json
```

**参数说明**（OmegaConf 命令行覆盖）：
- `model.name=<path>` —— 基础模型
- `model.num_latent_tokens=<int>` —— latent token 数 K（默认 3）
- `data.data_path=<path>` —— 训练数据
- `data.max_samples=<int>` —— 限制样本数（轻量调试用）
- `training.num_epochs=<int>` —— 训练轮数（默认 10，curriculum 推荐 15）
- `training.batch_size=<int>` —— 批大小（3090 推荐 2）
- `training.learning_rate=<float>` —— 学习率（默认 5e-5）
- `training.gradient_checkpointing=true` —— 启用梯度检查点省显存
- `training.use_lora=true` —— 启用 LoRA 微调
- `loss.transition_weight=<float>` —— 状态转移 loss 权重
- `loss.generation_weight=<float>` —— 生成 loss 权重
- `loss.anchor_weight=<float>` —— 锚点 loss 权重
- `loss.bridge_weight=<float>` —— 桥接 loss 权重
- `curriculum.enabled=true` —— 启用课程学习
- `checkpoint.save_dir=<path>` —— checkpoint 保存目录
- `checkpoint.keep_top_k=<int>` —— 保留最近 K 个 checkpoint（默认 3）

**轻量验证**（确认链路正常，2-3 分钟）：

```bash
python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/qwen2.5-math-1.5b \
    data.data_path=data/gsm8k_train.json \
    data.max_samples=200 \
    training.num_epochs=3 \
    training.batch_size=2 \
    logging.use_wandb=false
```

**训练监控**：通过 `tail -f checkpoints/stage1_transition/train.log` 查看分项 loss 趋势。判断 latent token 训练生效的标准：
- `transition` loss < 1.0
- `anchor` loss < 0.5
- `bridge` loss < 1.0
- `generation` loss < 0.5
- `transition` 和 `generation` **同步下降**

---

### 7. 完整诊断评估

```bash
python scripts/run_diagnostics.py \
    --checkpoint checkpoints/stage1_transition/final \
    --config config/exp/stage1_transition.yaml \
    --data_path data/gsm8k_test.json \
    --no_chat_template \
    --cot_accuracy 0.83 \
    --cot_avg_tokens 300 \
    --output results/diagnostics.json
```

**参数说明**：
- `--config`：训练配置（必需）
- `--checkpoint`：student 模型路径，可以是 `final/` 目录或 `.pt` 文件
- `--data_path`：评估数据
- `--split`：`test` / `train`（默认 `test`）
- `--batch_size`：评估批大小（默认 4）
- `--max_samples`：评估样本上限（0=全量，sanity check 用 10）
- `--max_new_tokens`：最大生成 token 数（默认 128）
- `--show_samples`：保存的定性样本数（默认 5）
- `--cot_accuracy`：Step 4 测得的 Teacher accuracy（用于计算 Retention）
- `--cot_avg_tokens`：Step 4 测得的 Teacher 平均输出 token 数（用于压缩比）
- `--direct_accuracy`：Direct Answer baseline accuracy（可选，计算 Relative Gain）
- `--no_chat_template`：**Student 评估必加**，禁用 chat template 以匹配训练时的 raw text 格式
- `--output`：结果 JSON 保存路径

**轻量 sanity check**：

```bash
python scripts/run_diagnostics.py \
    --checkpoint checkpoints/stage1_transition/final \
    --config config/exp/stage1_transition.yaml \
    --data_path data/gsm8k_test.json \
    --no_chat_template \
    --max_samples 10 \
    --show_samples 10 \
    --cot_accuracy 0.83 \
    --cot_avg_tokens 300 \
    --output results/sanity_check.json
```

---

### 8. 中断训练后手动导出 final（可选）

推荐使用脚本 `scripts/select_best_checkpoint.py`，自动扫描 train.log 找 val_loss 最低的 epoch 并导出：

```bash
# 自动选最优 epoch
python scripts/select_best_checkpoint.py \
    --ckpt_dir checkpoints/stage1_transition_v3 \
    --base_model models/qwen2.5-math-1.5b

# 手动指定某个 epoch
python scripts/select_best_checkpoint.py \
    --ckpt_dir checkpoints/stage1_transition_v3 \
    --base_model models/qwen2.5-math-1.5b \
    --epoch 4

# 批量处理所有消融实验
python scripts/select_best_checkpoint.py \
    --batch_root checkpoints/ablation \
    --base_model models/qwen2.5-math-1.5b
```

**参数说明**：
- `--ckpt_dir`：单个 checkpoint 目录
- `--batch_root`：批量处理根目录（适用于消融实验）
- `--epoch`：指定具体 epoch（默认自动选 val_loss 最低的）
- `--base_model`：基础模型路径
- `--num_latent_tokens`：latent token 数（默认 3）
- `--layer_ids`：对齐层（默认 -1 -2）
- `--no_lora`：跳过 LoRA 包装（如果模型未启用 LoRA）

**重要提示**：使用该脚本需训练时设 `checkpoint.keep_top_k=999`，避免最佳 epoch 被自动删除：

```bash
python scripts/train.py --config config/exp/stage1_transition.yaml \
    checkpoint.keep_top_k=999
```

---

## 训练数据格式

```json
{
  "question": "Tom has 5 apples...",
  "answer": "3",
  "steps": ["Tom starts with 5.", "He gives away 2.", "5 - 2 = 3."],
  "spans": [["Tom starts with 5.", "He gives away 2."], ["5 - 2 = 3."]]
}
```

---

## 4 个核心 Loss

| Loss | 作用 | 推荐权重 |
|------|------|---------|
| `transition_loss` | 对齐状态转移量 ΔH | 0.3-0.7 |
| `anchor_loss` | 防止 hidden state 漂移 | 0.05-0.2 |
| `bridge_loss` | 缓解 exposure mismatch | 0.05-0.15 |
| `generation_loss` | 答案生成交叉熵 | **0.2-0.4** |

> ⚠️ `generation_weight` 不要低于 0.2，否则模型无法学会输出答案格式。

---

## 诊断指标体系（§5.6）

`run_diagnostics.py` 一次性输出所有论文指标：

| 类别 | 指标 | 含义 |
|------|------|------|
| §5.6.2 性能 | Accuracy, Exact Match | 任务准确率 |
| §5.6.2 性能 | Accuracy Retention | 相对 CoT teacher 保留率 |
| §5.6.3 效率 | Avg Tokens, Token Reduction | 输出 token 数 / 压缩率 |
| §5.6.3 效率 | Compression Ratio | teacher_CoT_tokens / K |
| §5.6.3 效率 | Latency, Throughput | 端到端速度 |
| §5.6.4 对齐 | Transition Cosine | cos(ΔS, ΔT) 状态转移方向 |
| §5.6.4 对齐 | Normalized Transition Error | ‖ΔS-ΔT‖/‖ΔT‖ |
| §5.6.4 对齐 | Endpoint Drift | 边界状态绝对漂移 |
| §5.6.4 对齐 | Layer-wise CKA | 层间表征相似度 |
| §5.6.5 稳定性 | Collapse Rate | latent token 同质化检测 |
| §5.6.5 稳定性 | Pairwise Diversity | 表征多样性 |
| §5.6.5 稳定性 | Effective Rank | 表征有效秩 |

---

## 消融实验

```bash
# 默认：4 卡并行跑 12 个消融实验，每个 15 epoch + 自动评估
bash scripts/run_ablation.sh
```

**环境变量**（所有参数可通过环境变量覆盖）：

| 环境变量 | 默认值 | 含义 |
|----------|--------|------|
| `NUM_EPOCHS` | 15 | 训练轮数（快速验证可设 5）|
| `GPU_COUNT` | 4 | 并行卡数 |
| `BASE_MODEL` | `models/qwen2.5-math-1.5b` | 基础模型路径 |
| `TRAIN_DATA` | `data/gsm8k_train.json` | 训练数据 |
| `TEST_DATA` | `data/gsm8k_test.json` | 测试数据 |
| `COT_ACC` | 0.83 | Teacher CoT accuracy（用于 retention）|
| `COT_TOKENS` | 300 | Teacher 平均输出 token 数 |
| `SKIP_EVAL` | 0 | 设为 1 则只训练不评估 |

**常用示例**：

```bash
# 快速初筛（5 epoch，约 2.75 小时）
NUM_EPOCHS=5 bash scripts/run_ablation.sh

# 完整论文实验（15 epoch，约 8.25 小时）
bash scripts/run_ablation.sh

# 2 卡环境
GPU_COUNT=2 bash scripts/run_ablation.sh

# 只训练不自动评估
SKIP_EVAL=1 bash scripts/run_ablation.sh
```

**输出结构**：
- 检查点：`checkpoints/ablation/<name>/final/`
- 评估结果：`results/ablation/<name>.json`
- 训练日志：`checkpoints/ablation/<name>.train.log`
- 评估日志：`results/ablation/<name>.eval.log`

**消融配置说明**（`config/exp/ablation/` 下 12 个）：

| 配置 | 验证点 |
|------|--------|
| `no_transition.yaml` | 去掉 transition loss |
| `no_anchor.yaml` | 去掉 anchor loss |
| `no_bridge.yaml` | 去掉 bridge loss |
| `transition_only.yaml` | 只保留 transition loss |
| `full_loss.yaml` | 全部 4 个 loss |
| `k1.yaml` / `k2.yaml` / `k5.yaml` | 不同 latent token 数 K |
| `layer_last1.yaml` / `layer_last4.yaml` / `layer_all.yaml` | 不同 layer 选择 |
| `dist_cosine.yaml` | cosine 距离函数 |

---

## 对比基线

```bash
python scripts/run_baselines.py \
    --data_path data/gsm8k_train.json \
    --test_data data/gsm8k_test.json \
    --model_name models/qwen2.5-math-1.5b \
    --skip_training
```

**参数说明**：
- `--data_path`：训练数据（baseline 需要训练时用）
- `--test_data`：测试数据
- `--model_name`：基础模型路径
- `--skip_training`：跳过训练，只评估已有 checkpoint
- `--max_new_tokens`：最大生成 token 数（默认 128）
- `--batch_size`：评估批大小（默认 8）
- `--output`：结果 JSON 保存路径

> ⚠️ `run_baselines.py` 不支持 `--max_samples` 和 `--num_epochs` 参数。

---

## 超参搜索（4 卡并行示例）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/qwen2.5-math-1.5b data.data_path=data/gsm8k_train.json \
    training.learning_rate=1e-4 loss.transition_weight=0.5 \
    checkpoint.save_dir=checkpoints/search/lr1e4_tw05 &

CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/exp/stage1_transition.yaml \
    model.name=models/qwen2.5-math-1.5b data.data_path=data/gsm8k_train.json \
    training.learning_rate=5e-5 loss.transition_weight=0.5 \
    checkpoint.save_dir=checkpoints/search/lr5e5_tw05 &

wait
```

---

## 常见问题

**Q1：Accuracy = 0% 怎么排查？**

按顺序检查：
1. 评估时是否加了 `--no_chat_template`（Qwen 系列必加）
2. 训练日志中 `generation` loss 是否降到 0.5 以下
3. `latent_embeddings.pt` 是否在 `final/` 目录里
4. 用 `--show_samples 10` 看 predictions 是否合理

**Q2：训练 val_loss 早期开始上涨？**

属于过拟合。`keep_top_k=3` 会自动保留最近 3 个 checkpoint，可以回退到 val_loss 最低的那个手动导出 final（见 Step 8）。

**Q3：训练中断了怎么办？**

用现有 `.pt` checkpoint 手动构造 `final/`（见 Step 8）。

---

## 硬件需求

- 4× RTX 3090 24GB（单卡也可，调小 batch_size 即可）
- CUDA 12.2+
- Python 3.10+

## 技术栈

- PyTorch 2.0+
- HuggingFace Transformers + PEFT (LoRA)
- OmegaConf (配置管理)
- WandB (可选，实验追踪)
