# Latent Reasoning: State Transition Alignment

面向高效大模型推理的状态转移对齐式 Latent Reasoning Token 学习

## 核心思想

用少量 latent tokens 替代显式 Chain-of-Thought 推理链，通过对齐 Teacher 模型内部隐状态的**转移量（ΔH = h_{k+1} − h_k）**来训练 Student 模型在隐空间完成推理。

> **注意**：latent token 的作用是节省**推理阶段（prefill）**的 token 占用，而非减少输出 token 长度。
>
> latent token 是否真正生效，由 `num_latent_tokens > 0` **且**对应的隐状态对齐 loss 权重 > 0 共同决定，不能仅凭 generation loss 下降判断。

---

## 当前实验状态

| 实验 | 配置 | Accuracy | Avg Tokens | 状态 |
|------|------|:--------:|:----------:|:----:|
| **CoT SFT**（能力上界 / Teacher） | `gpt2_cot_sft.yaml` | 27.3% | 28.2 | ✅ 完成 |
| **Direct Answer**（无推理下界） | `gpt2_direct_answer.yaml` | 12.4% | 3.4 | ✅ 完成 |
| **Student（K=3, full method）** | `gpt2_student.yaml` | 训练中 | — | 🔄 |
| 消融 · no_trans（去 transition） | +CLI 覆盖 | 待评估 | — | ✅ 训完 |
| 消融 · task_only（纯 generation） | +CLI 覆盖 | 待评估 | — | ✅ 训完 |

- **基础模型**：GPT-2（`models/gpt2-local`，124.4M params）+ LoRA
- **训练数据**：GSM8k-Aug（`data/gsm8k_aug_train.json`，385K，纯表达式 CoT）
- **评估数据**：标准 GSM8K test（`data/gsm8k_test.json`，1319 条）

---

## 项目结构

```
src/
├── models/          # 模型封装(base) + 状态转移模块(state_transition) + 损失函数(loss_functions)
├── data/            # 数据加载(dataset) + 预处理(preprocessing) + Teacher状态提取(state_extractor)
├── training/        # 训练循环(trainer) + 课程学习(curriculum，已禁用)
└── eval/            # 评估器(evaluator) + 诊断指标(diagnostics) + 指标计算(metrics)

scripts/
├── download_gsm8k.py          # 下载 GSM8K 数据集
├── preprocess_data.py         # 数据预处理（生成 spans 字段）
├── train.py                   # 训练入口（CoT SFT / Direct Answer / Student 通用）
├── extract_teacher_states.py  # 提取 Teacher 隐状态到 HDF5
├── run_diagnostics.py         # 完整诊断评估（accuracy + §5.6 指标 + 定性样本）
├── evaluate.py                # 快速评估（不支持 --no_chat_template）
└── select_best_checkpoint.py  # 中断后手动导出最优 final

config/exp/                    # 全部自包含，无 defaults 继承
├── gpt2_cot_sft.yaml          # Stage 0: CoT SFT Teacher（num_latent=0）
├── gpt2_direct_answer.yaml    # 下界 baseline（mode=direct）
├── gpt2_student.yaml          # Stage 1: Student latent 训练（K=3）
├── extract.yaml               # Teacher states 提取
├── ablation/                  # 消融配置（旧，已被 CLI 覆盖方式取代，见下文）
└── baselines/                 # 对比基线配置
```

> **配置约定**：所有 YAML 完全自包含，不使用 `defaults` 继承；`curriculum.enabled: false` 全局禁用课程学习。

---

## 环境安装

```bash
conda create -n latent_reasoning python=3.10 -y
conda activate latent_reasoning
pip install -r requirements.txt
```

> 服务器访问 HuggingFace 受限时设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 完整流程（4 阶段）

### 1. 数据准备

训练用 GSM8k-Aug（385K，已在 `data/gsm8k_aug_train.json`），测试用标准 GSM8K test：

```bash
# 若需重新下载 test
python scripts/download_gsm8k.py --output data/gsm8k_test.json --split test
```

**预处理生成 spans 字段**（Student 训练与 Teacher 提取必需）：

```bash
python scripts/preprocess_data.py \
    --input data/gsm8k_aug_train.json \
    --output data/gsm8k_aug_train.json \
    --num_spans 3 --strategy fixed
```

- `--num_spans`：每条样本切分的 span 数 K（默认 3）
- `--strategy`：`fixed`（均匀合并 steps）/ `random` / `none`

> ⚠️ 数据已强制校验：`num_spans>0` 且 `strategy!=none` 时若缺 `spans` 字段会直接报错，不再静默 fallback。

---

### 2. Stage 0：CoT SFT Teacher 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/exp/gpt2_cot_sft.yaml
```

- 纯 generation loss（`num_latent=0`），模型学会显式 CoT 表达式推理
- 训练完成后自动导出最优 checkpoint 到 `checkpoints/gpt2/cot_sft/final`（LoRA adapter）

**评估 Teacher**（获取上界 accuracy）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_diagnostics.py \
    --config config/exp/gpt2_cot_sft.yaml \
    --checkpoint checkpoints/gpt2/cot_sft/final \
    --data_path data/gsm8k_test.json \
    --no_chat_template \
    --output results/cot_sft_eval.json
```

---

### 3. 提取 Teacher States（HDF5 缓存）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_teacher_states.py \
    --config config/exp/extract.yaml
```

- `extract.yaml` 自带 `teacher.teacher_path`、`output_dir`、`batch_size`，无需 CLI 传参
- 从 CoT SFT `final`（LoRA adapter）加载 teacher，在 span boundary 位置抓 `layer_ids=[-1,-2]` 的 hidden states
- 输出 `data/teacher_states_gpt2/teacher_states.h5`

> `num_latent_tokens: 0` 必须与 teacher 训练一致，否则 vocab size mismatch（50259 vs 50262）加载失败。

---

### 4. Stage 1：Student Latent 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config config/exp/gpt2_student.yaml \
    training.batch_size=32 training.gradient_accumulation_steps=4
```

- 输入格式：`Question ... <LATENT_0><LATENT_1><LATENT_2> Answer: <answer>`
- 4 个 loss 联合优化（transition + anchor + bridge + generation）
- **必须用 `batch_size=32`**：bridge loss 的二次前向占显存，默认 64 会 OOM（effective batch 仍=128）

**训练监控**（判断 latent token 是否生效）：

```bash
tail -f checkpoints/gpt2/student/train.log
```

| 分项 loss | 达标线 | 含义 |
|-----------|:------:|------|
| `transition` | < 1.0 | ΔH 转移量对齐 |
| `anchor` | < 0.5 | 绝对状态防漂移 |
| `bridge` | < 1.0 | 缓解 exposure mismatch |
| `generation` | < 0.5 | 答案生成质量 |

> latent token 真正生效的标志是 **transition/anchor/bridge 三个隐状态 loss 协同下降**；generation loss 仅反映 label 层面质量，不单独代表 latent 推理能力。

**评估 Student**：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_diagnostics.py \
    --config config/exp/gpt2_student.yaml \
    --checkpoint checkpoints/gpt2/student/final \
    --data_path data/gsm8k_test.json \
    --no_chat_template \
    --output results/student_k3_eval.json
```

---

### 5. Direct Answer 下界（可选，独立跑）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/exp/gpt2_direct_answer.yaml
```

- `mode=direct`，训练数据纯净 Q→A，不含任何 CoT
- **LR=5e-4**（比 3e-3 低）：答案监督极稀疏（1-3 token），高 LR 会震荡爆炸

---

## 关键超参（对齐 CODI 论文）

| 类别 | 参数 | 值 |
|------|------|-----|
| 模型 | 基础模型 | GPT-2 (`models/gpt2-local`) |
| 模型 | num_latent_tokens (Student) | 3 |
| 模型 | layer_ids | [-1, -2] |
| LoRA | r / alpha | 128 / 32 |
| LoRA | target_modules | `["c_attn", "c_proj"]`（GPT-2 层名） |
| LoRA | dropout | 0.0 |
| 训练 | 精度 | **bf16**（fp16 会 NaN，见下文） |
| 训练 | learning_rate | 3e-3（CoT SFT / Student）、5e-4（Direct Answer） |
| 训练 | effective batch | 128 |
| 训练 | num_epochs | 40 |
| 训练 | warmup_ratio / weight_decay / max_grad_norm | 0.03 / 0.1 / 2.0 |
| 训练 | seed | 11 |
| 数据 | num_spans / span_strategy | 3 / fixed |

---

## 4 个核心 Loss

| Loss | 作用对象 | 计算 | 权重 |
|------|---------|------|:----:|
| `transition_loss` | latent boundary 的 ΔH | smooth_l1（除以 teacher std 归一化） | 0.5 |
| `anchor_loss` | latent boundary 绝对状态 | smooth_l1 | 0.1 |
| `bridge_loss` | student vs teacher-prefix rollout | 二次前向对比 | 0.1 |
| `generation_loss` | answer token | cross entropy | 0.3 |

> - `transition_loss` 使用 **smooth_l1**（梯度有界），**不是 MSE**（会导致 NaN）。
> - `normalize_transition: true` 对应 CODI `--distill_loss_div_std True`：ΔH 除以 batch std，均衡各层贡献。
> - `bridge_weight` 必须 > 0，不得因显存移除——它是能力对齐的关键设计。

---

## 训练数据格式

GSM8k-Aug（纯表达式 CoT，预处理后带 spans）：

```json
{
  "question": "Out of 600 employees, 30% got promoted...",
  "answer": "360",
  "steps": ["<<600*30/100=180>>", "<<600*10/100=60>>", "<<180+60=240>>", "<<600-240=360>>"],
  "spans": [
    ["<<600*30/100=180>>", "<<600*10/100=60>>"],
    ["<<180+60=240>>"],
    ["<<600-240=360>>"]
  ]
}
```

> `steps` 字段须原样保留含 `<<>>` 的表达式列表；`fixed` 策略把 steps 均匀合并到 K 个 span，不丢信息。

---

## 消融实验

> 旧的 `config/exp/ablation/*.yaml` 仍继承已删除的 `stage1_transition`，**已失效**。当前统一用 `gpt2_student.yaml` + CLI 覆盖 loss 权重和 save_dir。

```bash
# 去掉 transition loss
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config config/exp/gpt2_student.yaml \
    training.batch_size=32 training.gradient_accumulation_steps=4 \
    loss.transition_weight=0.0 loss.anchor_weight=0.1 loss.bridge_weight=0.1 loss.generation_weight=0.3 \
    checkpoint.save_dir=checkpoints/gpt2/ablation_no_trans \
    logging.run_name=ablation-no-transition

# 纯 generation（task only，latent token 无隐状态对齐监督）
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --config config/exp/gpt2_student.yaml \
    training.batch_size=32 training.gradient_accumulation_steps=4 \
    loss.transition_weight=0.0 loss.anchor_weight=0.0 loss.bridge_weight=0.0 loss.generation_weight=1.0 \
    checkpoint.save_dir=checkpoints/gpt2/ablation_task_only \
    logging.run_name=ablation-task-only
```

评估消融模型（架构与 student 一致，直接用 `gpt2_student.yaml` 作 config）：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_diagnostics.py \
    --config config/exp/gpt2_student.yaml \
    --checkpoint checkpoints/gpt2/ablation_no_trans/final \
    --data_path data/gsm8k_test.json \
    --no_chat_template \
    --output results/ablation/no_trans.json
```

> ⚠️ **不同配置的 val_loss 不可直接比较**：task_only 只有 generation 一项，no_trans 有 3 项，full 有 4 项，分量越多总 val_loss 越大。对比必须看评估出的 **accuracy**。

---

## 诊断指标体系（§5.6）

`run_diagnostics.py` 一次性输出：

| 类别 | 指标 |
|------|------|
| 性能 | Accuracy、Accuracy Retention（相对 CoT teacher） |
| 效率 | Avg Tokens、Token Reduction、Compression Ratio、Latency、Throughput |
| 对齐 | Transition Cosine、Normalized Transition Error、Endpoint Drift、Layer-wise CKA |
| 稳定性 | Collapse Rate、Pairwise Diversity、Effective Rank |

Stage 0（无 latent token）评估时会自动跳过 transition 相关指标。

---

## 显存优化与训练稳定性

单卡 24GB（RTX 3090/4090）跑 Student 训练的关键实践：

| 问题 | 根因 | 方案 |
|------|------|------|
| Teacher states OOM | 全量 385K×K×2层×768 ≈ 6.6GB，加上模型+bridge 二次前向放不下 | teacher states **CPU 存储 + fp16**，按 batch 索引切片后 `.to(device)`（每 step 仅搬 ~74KB） |
| `torch.stack` 反复分配 | 每 step 对全量 tensor 做 stack | 首次调用缓存 `_cached_boundary_end` 等，后续复用 |
| loss=NaN（fp16） | fp16 数值范围 ±65504，LR=3e-3 溢出 | 用 **bf16**（范围同 fp32） |
| loss=NaN（loss 函数） | transition 用 MSE + fp16/bf16 混合 | 改 **smooth_l1** + teacher 张量 `.to(student.dtype)` 统一精度 |
| Student batch 放不下 | bridge loss 二次前向使激活翻倍 | `batch_size=32, gradient_accumulation_steps=4`（effective 仍=128） |

---

## 中断恢复 / 手动导出 final

训练时已设 `keep_top_k: 999` 保留全部 epoch checkpoint。中断后自动导出最优：

```bash
python scripts/select_best_checkpoint.py \
    --ckpt_dir checkpoints/gpt2/student \
    --base_model models/gpt2-local \
    --num_latent_tokens 3 \
    --layer_ids -1 -2
```

> `--num_latent_tokens` 必须与训练一致，否则 vocab size mismatch。

---

## 常见问题

**Q1：Accuracy = 0% 怎么排查？**
1. 评估是否加了 `--no_chat_template`（训练是 raw text `Question:...\nAnswer:...`，不加会走 chat template 导致格式不匹配）
2. 是否用了 `final/` 目录而非 `checkpoint_best.pt`（LoRA 评估必须用 final）
3. 训练日志 `generation` loss 是否降到合理水平
4. `--show_samples 10` 看 predictions 是否合理

**Q2：loss 突然变 NaN？**
确认用的是 `bf16: true` 而非 `fp16`。GPT-2 + LR=3e-3 在 fp16 下会数值溢出。

**Q3：CUDA OOM？**
Student 训练用 `batch_size=32`；teacher states 保持 CPU 存储；确认没有其他进程（如 VS Code fileWatcher）占满 RAM。

**Q4：evaluate.py 报 unrecognized arguments: --no_chat_template？**
`evaluate.py` 不支持该参数，Student/CoT SFT 评估请用 `run_diagnostics.py`。

---

## 硬件需求与技术栈

- **硬件**：RTX 3090/4090 24GB（单卡可跑；4 卡可并行跑不同实验/消融）
- **技术栈**：PyTorch 2.0+、HuggingFace Transformers + PEFT (LoRA)、OmegaConf、h5py
