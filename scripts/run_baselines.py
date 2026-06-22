#!/usr/bin/env python3
"""运行所有基线实验并输出对比表格。

Usage:
    python scripts/run_baselines.py --data_path data/gsm8k_train.json --model_name gpt2
    python scripts/run_baselines.py --skip_training --test_data data/gsm8k_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.data.dataset import LatentReasoningDataset, collate_fn, load_gsm8k
from src.eval.evaluator import Evaluator
from src.models.base import LatentReasoningModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Baseline Definitions ----
BASELINES = {
    "Direct Answer": {
        "config": "config/exp/baselines/direct_answer.yaml",
        "checkpoint": "checkpoints/baselines/direct_answer/final",
        "needs_training": True,
    },
    "Explicit CoT": {
        "config": "config/exp/baselines/explicit_cot.yaml",
        "checkpoint": "checkpoints/stage0_cot/final",
        "needs_training": False,  # 直接使用 Stage 0 teacher
    },
    "Short CoT": {
        "config": "config/exp/baselines/short_cot.yaml",
        "checkpoint": "checkpoints/baselines/short_cot/final",
        "needs_training": True,
    },
}

# Our method (latent reasoning with K=3)
OURS = {
    "Ours (K=3)": {
        "config": "config/exp/stage1_transition.yaml",
        "checkpoint": "checkpoints/stage1_transition/final",
        "needs_training": False,  # 假设已训练完成
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments and compare")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/gsm8k_train.json",
        help="Training data path.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/gsm8k_test.json",
        help="Test data path for evaluation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only run evaluation on existing checkpoints.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate during evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON.",
    )
    return parser.parse_args()


def train_baseline(config_path: str, model_name: str, data_path: str) -> None:
    """Train a baseline model using the training script."""
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", config_path,
        f"model.name={model_name}",
        f"data.data_path={data_path}",
        "logging.use_wandb=false",
    ]
    logger.info("Training: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Training failed for config: %s", config_path)
        raise RuntimeError(f"Training failed: {config_path}")


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    test_data_path: str,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> dict:
    """Evaluate a model and return metrics including avg token count."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config for model params
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        base_refs = cfg.pop("defaults")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        for ref in base_refs:
            base_path = os.path.join(config_dir, ref + ".yaml")
            if os.path.exists(base_path):
                base_cfg = OmegaConf.load(base_path)
                cfg = OmegaConf.merge(base_cfg, cfg)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Load model
    model = LatentReasoningModel(
        model_name=checkpoint_path,
        layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
        num_latent_tokens=model_cfg.get("num_latent_tokens", 0),
        device=device,
    )

    # Load test data
    raw_data = load_gsm8k(data_path=test_data_path, split="test")

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 0),
        span_strategy=data_cfg.get("span_strategy", "none"),
    )

    from functools import partial
    from torch.utils.data import DataLoader

    pad_token_id = model.tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
    )

    # Evaluate
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        metrics=["accuracy", "exact_match"],
        max_new_tokens=max_new_tokens,
    )

    results = evaluator.evaluate()

    # Compute average output token count
    avg_tokens = results.get("avg_output_tokens", max_new_tokens)

    return {
        "accuracy": results.get("accuracy", 0.0),
        "avg_tokens": avg_tokens,
    }


def compute_reduction(avg_tokens: float, cot_tokens: float) -> float:
    """Compute token reduction ratio relative to explicit CoT."""
    if cot_tokens == 0:
        return 0.0
    return (1.0 - avg_tokens / cot_tokens) * 100.0


def print_comparison_table(all_results: dict) -> str:
    """Print a Markdown-formatted comparison table."""
    # Get explicit CoT token count as reference
    cot_tokens = all_results.get("Explicit CoT", {}).get("avg_tokens", 200)

    header = "| Method         | GSM8K Acc | Avg Tokens | Reduction |"
    separator = "|----------------|-----------|------------|-----------|"
    rows = [header, separator]

    for method, res in all_results.items():
        acc = res.get("accuracy", 0.0)
        avg_tok = res.get("avg_tokens", 0)
        reduction = compute_reduction(avg_tok, cot_tokens)
        row = f"| {method:<14} | {acc*100:>7.1f}% | ~{avg_tok:<9.0f}| {reduction:>7.1f}%  |"
        rows.append(row)

    table = "\n".join(rows)
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print(table)
    print("=" * 60 + "\n")
    return table


def main() -> None:
    args = parse_args()

    all_results = {}

    # ---- Step 1: Train baselines (if needed) ----
    if not args.skip_training:
        for name, info in BASELINES.items():
            if info["needs_training"]:
                checkpoint_dir = info["checkpoint"]
                if not os.path.exists(checkpoint_dir):
                    logger.info("Training baseline: %s", name)
                    train_baseline(
                        config_path=info["config"],
                        model_name=args.model_name,
                        data_path=args.data_path,
                    )
                else:
                    logger.info("Checkpoint exists for %s, skipping training.", name)

    # ---- Step 2: Evaluate all methods ----
    all_methods = {**BASELINES, **OURS}
    for name, info in all_methods.items():
        checkpoint_path = info["checkpoint"]
        if not os.path.exists(checkpoint_path):
            logger.warning(
                "Checkpoint not found for %s at %s, skipping evaluation.",
                name,
                checkpoint_path,
            )
            # Use placeholder results
            all_results[name] = {"accuracy": 0.0, "avg_tokens": 0}
            continue

        logger.info("Evaluating: %s", name)
        try:
            results = evaluate_model(
                checkpoint_path=checkpoint_path,
                config_path=info["config"],
                test_data_path=args.test_data,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )
            all_results[name] = results
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", name, e)
            all_results[name] = {"accuracy": 0.0, "avg_tokens": 0}

    # ---- Step 3: Print comparison table ----
    table = print_comparison_table(all_results)

    # ---- Step 4: Save results ----
    output_path = args.output or "results/baseline_comparison.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"results": all_results, "table": table},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
