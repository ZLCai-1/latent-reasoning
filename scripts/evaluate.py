#!/usr/bin/env python3
"""
Evaluation script for Latent Reasoning models.

Usage:
    python scripts/evaluate.py \
        --config config/exp/stage1_transition.yaml \
        --checkpoint checkpoints/stage1_transition/final \
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import LatentReasoningDataset, collate_fn, load_gsm8k
from src.eval.evaluator import Evaluator
import warnings

from src.models.base import LatentReasoningModel

# GPT-2 uses absolute positional embeddings; right-padding + attention_mask is correct
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Latent Reasoning Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=5,
        help="Number of qualitative samples to display.",
    )
    parser.add_argument(
        "--cot_baseline_tokens",
        type=int,
        default=200,
        help="Expected avg token count for explicit CoT baseline (for reduction ratio).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to local data file (overrides config data.data_path).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Resolve base config if needed
    if "defaults" in cfg:
        base_refs = cfg.pop("defaults")
        config_dir = os.path.dirname(os.path.abspath(args.config))
        for ref in base_refs:
            base_path = os.path.join(config_dir, ref + ".yaml")
            if os.path.exists(base_path):
                base_cfg = OmegaConf.load(base_path)
                cfg = OmegaConf.merge(base_cfg, cfg)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model = LatentReasoningModel(
        model_name=args.checkpoint,
        layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
        device=device,
    )

    # Load data (CLI --data_path overrides config)
    data_path = args.data_path or data_cfg.get("data_path")
    raw_data = load_gsm8k(
        data_path=data_path,
        split=args.split,
    )

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
    )

    pad_token_id = model.tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id, padding_side="right"),
    )

    # Evaluate
    metrics = eval_cfg.get("metrics", ["accuracy", "exact_match"])
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        metrics=list(metrics),
        max_new_tokens=args.max_new_tokens,
        cot_baseline_tokens=args.cot_baseline_tokens,
    )

    results = evaluator.evaluate()
    logger.info("Evaluation results: %s", results)

    # Print formatted evaluation summary
    print("\n" + "=" * 60)
    print("=== Evaluation Results ===")
    print("=" * 60)
    if "accuracy" in results:
        print(f"Accuracy: {results['accuracy'] * 100:.1f}%")
    if "exact_match" in results:
        print(f"Exact Match: {results['exact_match'] * 100:.1f}%")
    if "avg_tokens" in results:
        cot_bl = int(results.get("cot_baseline_tokens", 200))
        print(f"Avg Output Tokens: {results['avg_tokens']:.0f} (vs CoT baseline ~{cot_bl})")
    if "token_reduction" in results:
        print(f"Token Reduction: {results['token_reduction'] * 100:.1f}%")
    if "avg_latency_ms" in results:
        print(f"Avg Latency: {results['avg_latency_ms']:.0f}ms/sample")
    print("=" * 60)

    # Show qualitative samples
    if args.show_samples > 0:
        samples = evaluator.generate_samples(num_samples=args.show_samples)
        print("\n" + "=" * 60)
        print("QUALITATIVE SAMPLES")
        print("=" * 60)
        for i, s in enumerate(samples):
            print(f"\n--- Sample {i + 1} ---")
            print(f"Input:      {s['input'][:200]}…")
            print(f"Prediction: {s['prediction'][:200]}")
            print(f"Reference:  {s['reference'][:200]}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
