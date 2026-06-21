#!/usr/bin/env python
"""
Data Preprocessing Script for Latent Reasoning.

Reads raw data (question/answer/steps) and outputs:
1. JSON with pre-computed spans (中间产物)
2. 可选：tokenized training-ready data (.pt) 包含 input_ids, labels, boundary_positions

Usage:
    # 只做 span 划分
    python scripts/preprocess_data.py \
        --input data/mini_gsm8k.json \
        --output data/processed.json \
        --num_spans 2 \
        --strategy fixed

    # 同时输出 tokenized 训练数据（可直接看到喂给模型的东西）
    python scripts/preprocess_data.py \
        --input data/mini_gsm8k.json \
        --output data/processed.json \
        --num_spans 2 \
        --strategy fixed \
        --tokenize \
        --model gpt2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.data.preprocessing import split_into_spans, prepare_training_sample


def _load_tokenizer(model_name: str):
    """加载 tokenizer，网络不可用时回退到本地构建。"""
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except (OSError, RuntimeError):
        print(f"  [WARNING] 无法下载 '{model_name}'，使用本地构建的 mini tokenizer")
        return _build_local_tokenizer()


def _build_local_tokenizer():
    """构建离线可用的 mini BPE tokenizer（无需网络）。"""
    import tempfile
    import json as _json
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2TokenizerFast

    base_tokenizer = Tokenizer(models.BPE())
    base_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer_obj = trainers.BpeTrainer(
        vocab_size=5000,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        min_frequency=1,
    )
    corpus = [
        "Tom has 5 apples. He gives 2 to Mary. How many does he have?",
        "A store has 10 books. They sell 3 in the morning.",
        "Sarah has 8 candies. She eats 3 and gives 2 to her friend.",
        "Question: Answer: The total is 5 - 2 = 3.",
        "A farmer has 12 chickens. He buys 5 more and then sells 4.",
        "Lisa has 20 dollars. She spends 7 on lunch and 5 on a book.",
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    ]
    base_tokenizer.train_from_iterator(corpus, trainer=trainer_obj)

    tmpdir = tempfile.mkdtemp()
    base_tokenizer.save(os.path.join(tmpdir, "tokenizer.json"))
    with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
        _json.dump({"model_type": "gpt2", "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>", "pad_token": "<|padding|>"}, f)
    with open(os.path.join(tmpdir, "special_tokens_map.json"), "w") as f:
        _json.dump({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
                    "pad_token": "<|padding|>"}, f)

    tokenizer = GPT2TokenizerFast.from_pretrained(tmpdir)
    tokenizer.pad_token = "<|padding|>"
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute spans for latent reasoning data."
    )
    parser.add_argument("--input", required=True, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--num_spans", type=int, default=3, help="Number of spans (K). Default: 3."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=["fixed", "random", "semantic"],
        help="Span splitting strategy. Default: fixed.",
    )
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Also tokenize and output training-ready .pt file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name for tokenizer (used with --tokenize). Default: gpt2.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization. Default: 512.",
    )
    args = parser.parse_args()

    # Load input data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[preprocess_data] Loaded {len(data)} records from {args.input}")
    print(f"[preprocess_data] Strategy: {args.strategy}, num_spans: {args.num_spans}")

    # ================================================================
    # Stage 1: Span 划分
    # ================================================================
    processed = []
    total_spans = 0
    for record in data:
        steps = record.get("steps", [])
        spans = split_into_spans(steps, num_spans=args.num_spans, strategy=args.strategy)
        new_record = {
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "steps": steps,
            "spans": spans,
        }
        processed.append(new_record)
        total_spans += len(spans)

    # Write span-processed JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    avg_spans = total_spans / len(processed) if processed else 0
    print(f"[preprocess_data] Wrote {len(processed)} records to {args.output}")
    print(f"[preprocess_data] Average spans per sample: {avg_spans:.1f}")

    # ================================================================
    # Stage 2: Tokenize（可选，--tokenize 时执行）
    # ================================================================
    if not args.tokenize:
        print(f"[preprocess_data] Done. (加 --tokenize 可输出训练数据)")
        return

    print(f"\n[tokenize] Loading tokenizer: {args.model}")
    from src.models.base import SPECIAL_TOKENS

    tokenizer = _load_tokenizer(args.model)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    print(f"[tokenize] Vocab size: {len(tokenizer)}")
    print(f"[tokenize] <SPAN_START> id: {tokenizer.convert_tokens_to_ids('<SPAN_START>')}")
    print(f"[tokenize] <SPAN_END>   id: {tokenizer.convert_tokens_to_ids('<SPAN_END>')}")

    # Tokenize each sample
    tokenized_samples = []
    for i, record in enumerate(processed):
        sample = prepare_training_sample(
            question=record["question"],
            spans=record["spans"],
            answer=record["answer"],
            tokenizer=tokenizer,
            num_spans=args.num_spans,
            span_strategy=args.strategy,
            max_seq_length=args.max_seq_length,
        )
        tokenized_samples.append(sample)

    # Save as .pt
    pt_output = args.output.rsplit(".", 1)[0] + "_tokenized.pt"
    torch.save(tokenized_samples, pt_output)
    print(f"[tokenize] Saved {len(tokenized_samples)} tokenized samples to {pt_output}")

    # ================================================================
    # 打印前 3 条样本的详细内容，方便直接查看
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  训练数据预览（前 {min(3, len(tokenized_samples))} 条）")
    print(f"{'=' * 70}")

    span_start_id = tokenizer.convert_tokens_to_ids("<SPAN_START>")
    span_end_id = tokenizer.convert_tokens_to_ids("<SPAN_END>")

    for i, sample in enumerate(tokenized_samples[:3]):
        print(f"\n{'─' * 70}")
        print(f"  Sample {i}")
        print(f"{'─' * 70}")

        input_ids = sample["input_ids"]
        labels = sample["labels"]
        bp = sample.get("boundary_positions", None)

        # 原始文本
        print(f"\n  [原始] question: {processed[i]['question']}")
        print(f"  [原始] spans:    {processed[i]['spans']}")
        print(f"  [原始] answer:   {processed[i]['answer']}")

        # 拼接后的完整文本（decode 回来看）
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"\n  [拼接文本] {full_text}")

        # input_ids
        print(f"\n  [input_ids] shape={list(input_ids.shape)}, 前30个token:")
        ids_list = input_ids.tolist()
        tokens_preview = tokenizer.convert_ids_to_tokens(ids_list[:30])
        for j in range(min(30, len(ids_list))):
            marker = ""
            if ids_list[j] == span_start_id:
                marker = " ◀ SPAN_START"
            elif ids_list[j] == span_end_id:
                marker = " ◀ SPAN_END"
            print(f"    pos={j:3d}  id={ids_list[j]:5d}  token='{tokens_preview[j]}'{marker}")
        if len(ids_list) > 30:
            print(f"    ... 共 {len(ids_list)} 个 token")

        # labels（哪些位置参与 loss）
        labels_list = labels.tolist()
        masked_count = sum(1 for x in labels_list if x == -100)
        active_count = len(labels_list) - masked_count
        print(f"\n  [labels] 总长={len(labels_list)}, 掩码(-100)={masked_count}, 参与loss={active_count}")
        # 找到第一个非-100的位置
        first_active = next((j for j, x in enumerate(labels_list) if x != -100), None)
        if first_active is not None:
            print(f"    loss 从 position {first_active} 开始")
            print(f"    loss 区域对应文本: {tokenizer.decode([x for x in labels_list if x != -100])}")

        # boundary_positions
        if bp is not None:
            print(f"\n  [boundary_positions] {bp.tolist()}")
            print(f"    共 {len(bp)} 个边界位置，对应 token:")
            for j, pos in enumerate(bp.tolist()):
                token_at_pos = tokenizer.convert_ids_to_tokens([ids_list[pos]])[0]
                print(f"      位置 {pos:3d} → '{token_at_pos}'")
        else:
            print(f"\n  [boundary_positions] 无（span_strategy='none' 时不产生）")

    print(f"\n{'=' * 70}")
    print(f"  Done. 训练时 DataLoader 直接读取这些 tensor。")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
