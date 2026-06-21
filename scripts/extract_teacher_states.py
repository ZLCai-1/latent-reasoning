#!/usr/bin/env python3
"""
Extract and cache teacher model hidden states at span boundaries.

This script should be run BEFORE Stage 1 training.  It loads the
teacher model (e.g. a Stage 0 trained checkpoint), processes the
dataset, and saves boundary hidden states + transition vectors to
an HDF5 file.

Usage:
    python scripts/extract_teacher_states.py \
        --config config/exp/stage1_transition.yaml \
        --teacher_path checkpoints/stage0_cot/final \
        --output_dir teacher_states
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf

from src.data.dataset import LatentReasoningDataset, collate_fn, load_gsm8k
from src.data.state_extractor import TeacherStateExtractor
from src.models.base import LatentReasoningModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract teacher hidden states for transition alignment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/exp/stage1_transition.yaml",
        help="Config file (used for data / model settings).",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=None,
        help="Path to the teacher model checkpoint.  Overrides config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="teacher_states",
        help="Directory for the HDF5 cache.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for extraction.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples to extract (0 = all).",
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
    teacher_cfg = cfg.get("teacher", {})

    # Teacher model name
    teacher_name = args.teacher_path or teacher_cfg.get("model_name") or model_cfg.get("name", "gpt2")
    layer_ids = list(model_cfg.get("layer_ids", [-1, -2]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Teacher model: %s", teacher_name)
    logger.info("Layer IDs: %s", layer_ids)
    logger.info("Output dir: %s", args.output_dir)

    # Load a temporary model just to get the tokenizer
    temp_model = LatentReasoningModel(
        model_name=teacher_name,
        layer_ids=layer_ids,
        device=device,
    )

    # Load dataset
    raw_data = load_gsm8k(
        data_path=data_cfg.get("data_path"),
        split="train",
    )

    if args.max_samples and args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]
        logger.info("Using max_samples=%d", args.max_samples)

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=temp_model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
    )

    # Clean up temp model to free memory
    del temp_model
    torch.cuda.empty_cache()

    # Create extractor and run
    extractor = TeacherStateExtractor(
        model_name=teacher_name,
        layer_ids=layer_ids,
        device=device,
        cache_dir=args.output_dir,
        store_fp16=teacher_cfg.get("store_fp16", True),
    )

    cache_path = extractor.extract_and_cache(
        dataset=dataset,
        batch_size=args.batch_size,
        cache_filename="teacher_states.h5",
    )

    logger.info("Extraction complete.  Cache saved to: %s", cache_path)


if __name__ == "__main__":
    main()
