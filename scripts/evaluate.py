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
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
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
        default=10,
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max number of samples to evaluate (0 = all).",
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
    # For LoRA checkpoints: load base model first, then apply LoRA adapter
    checkpoint_path = args.checkpoint
    base_model_name = model_cfg.get("name", "gpt2")
    num_latent_tokens = model_cfg.get("num_latent_tokens", 0)

    # Check if checkpoint is a LoRA adapter (has adapter_config.json)
    is_lora_checkpoint = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    if is_lora_checkpoint:
        logger.info("Loading base model from %s + LoRA from %s", base_model_name, checkpoint_path)
        model = LatentReasoningModel(
            model_name=base_model_name,
            layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
            num_latent_tokens=num_latent_tokens,
            device=device,
        )
        # Load LoRA adapter
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, checkpoint_path)
        model.model.eval()
        # Load latent_embeddings from checkpoint if saved separately
        latent_path = os.path.join(checkpoint_path, "latent_embeddings.pt")
        if os.path.exists(latent_path):
            latent_state = torch.load(latent_path, map_location=device)
            model.latent_embeddings.load_state_dict(latent_state)
            logger.info("Loaded latent_embeddings from %s", latent_path)
        # Try loading from pytorch checkpoint
        else:
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_epoch') and f.endswith('.pt')]
            if not ckpt_files:
                # Check parent dir for latest checkpoint
                parent = os.path.dirname(checkpoint_path)
                ckpt_files = sorted([f for f in os.listdir(parent) if f.startswith('checkpoint_epoch') and f.endswith('.pt')])
                if ckpt_files:
                    ckpt_path = os.path.join(parent, ckpt_files[-1])
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    if 'model_state_dict' in ckpt and 'latent_embeddings.weight' in ckpt['model_state_dict']:
                        model.latent_embeddings.weight.data = ckpt['model_state_dict']['latent_embeddings.weight']
                        logger.info("Loaded latent_embeddings from %s", ckpt_path)
    else:
        logger.info("Loading model from %s", checkpoint_path)
        model = LatentReasoningModel(
            model_name=checkpoint_path,
            layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
            num_latent_tokens=num_latent_tokens,
            device=device,
        )

    # Load data (CLI --data_path overrides config)
    data_path = args.data_path or data_cfg.get("data_path")
    raw_data = load_gsm8k(
        data_path=data_path,
        split=args.split,
    )

    # Truncate data if --max_samples specified
    if args.max_samples and args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]
        logger.info("Evaluating first %d samples only", args.max_samples)

    # Determine mode based on num_latent_tokens
    num_latent_tokens = model_cfg.get("num_latent_tokens", 0)

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
        mode="student" if num_latent_tokens > 0 else "teacher",
        num_latent_tokens=num_latent_tokens,
    )

    pad_token_id = model.tokenizer.pad_token_id or 0
    # Auto-detect padding side: GPT-2 (absolute pos) needs right, others (RoPE) need left
    model_type = getattr(model.model.config, 'model_type', 'gpt2')
    eval_padding_side = "right" if model_type == "gpt2" else "left"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id, padding_side=eval_padding_side),
    )

    # Evaluate
    metrics = eval_cfg.get("metrics", ["accuracy"])
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

    if "avg_tokens" in results:
        cot_bl = int(results.get("cot_baseline_tokens", 200))
        print(f"Avg Output Tokens: {results['avg_tokens']:.0f} (vs CoT baseline ~{cot_bl})")
    if "token_reduction" in results:
        print(f"Token Reduction: {results['token_reduction'] * 100:.1f}%")
    if "avg_latency_ms" in results:
        print(f"Avg Latency: {results['avg_latency_ms']:.0f}ms/sample")
    print("=" * 60)

    # Show qualitative samples
    samples = []
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
        output_data = {
            "metrics": results,
            "samples": samples,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
