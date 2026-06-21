#!/usr/bin/env python3
"""
下载 GSM8K 数据集并转为项目格式。

Usage:
    python scripts/download_gsm8k.py --output data/gsm8k_raw.json --split train
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_steps(solution: str) -> list:
    """从 GSM8K 的 solution 字段提取推理步骤。"""
    # GSM8K 的 solution 用换行分隔步骤，最后一行是 #### 答案
    lines = [l.strip() for l in solution.strip().split("\n") if l.strip()]
    steps = [l for l in lines if not l.startswith("####")]
    return steps


def extract_answer(solution: str) -> str:
    """从 GSM8K 的 solution 字段提取最终答案。"""
    match = re.search(r"####\s*(.+)", solution)
    if match:
        return match.group(1).strip()
    # fallback: 最后一行
    lines = [l.strip() for l in solution.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def main():
    parser = argparse.ArgumentParser(description="下载 GSM8K 并转为项目格式")
    parser.add_argument("--output", type=str, default="data/gsm8k_raw.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="下载哪个 split")
    args = parser.parse_args()

    print(f"正在从 HuggingFace 下载 GSM8K ({args.split} split)...")
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    print(f"下载完成，共 {len(dataset)} 条数据")
    print("正在转换为项目格式...")

    records = []
    for item in dataset:
        question = item["question"]
        solution = item["answer"]
        steps = extract_steps(solution)
        answer = extract_answer(solution)

        records.append({
            "question": question,
            "answer": answer,
            "steps": steps,
        })

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"完成！保存到 {args.output}")
    print(f"  样本数: {len(records)}")
    print(f"  平均步骤数: {sum(len(r['steps']) for r in records) / len(records):.1f}")
    print(f"\n示例 (第1条):")
    print(f"  问题: {records[0]['question'][:80]}...")
    print(f"  步骤数: {len(records[0]['steps'])}")
    print(f"  答案: {records[0]['answer']}")


if __name__ == "__main__":
    main()
