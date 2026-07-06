#!/usr/bin/env python3
"""
Main training entry point for Latent Reasoning.

Usage:
    # Stage 0: Train CoT teacher
    python scripts/train.py --config config/exp/stage0_cot.yaml

    # Stage 1: Transition alignment
    python scripts/train.py --config config/exp/stage1_transition.yaml

    # Override via CLI
    python scripts/train.py --config config/base.yaml \
        training.learning_rate=1e-4 \
        training.batch_size=8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functools import partial

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from src.data.dataset import LatentReasoningDataset, collate_fn, load_gsm8k
from src.models.base import LatentReasoningModel
from src.models.state_transition import StateTransitionModule
from src.training.curriculum import CurriculumScheduler
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent Reasoning Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides (e.g. training.lr=1e-4).",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge YAML config with CLI overrides."""
    cfg = OmegaConf.load(args.config)

    # If the config references a base via 'defaults', merge it
    if "defaults" in cfg:
        base_refs = cfg.pop("defaults")
        config_dir = os.path.dirname(os.path.abspath(args.config))
        for ref in base_refs:
            base_path = os.path.join(config_dir, ref + ".yaml")
            if os.path.exists(base_path):
                base_cfg = OmegaConf.load(base_path)
                cfg = OmegaConf.merge(base_cfg, cfg)

    # Apply CLI overrides
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # Auto-save logs to checkpoint directory
    log_dir = cfg.get("checkpoint", {}).get("save_dir", "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s \u2014 %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", log_file)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ---- Model ----
    model_cfg = cfg.get("model", {})
    num_latent_tokens = model_cfg.get("num_latent_tokens", 3)
    logger.info("num_latent_tokens=%d", num_latent_tokens)

    model = LatentReasoningModel(
        model_name=model_cfg.get("name", "gpt2"),
        layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
        num_latent_tokens=num_latent_tokens,
        device=device,
    )

    # Enable gradient checkpointing to save GPU memory
    train_cfg_tmp = cfg.get("training", {})
    if train_cfg_tmp.get("gradient_checkpointing", False):
        model.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA if configured (saves memory vs full fine-tuning)
    if train_cfg_tmp.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        lora_cfg = cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.model = get_peft_model(model.model, lora_config)
        model.model.print_trainable_parameters()
        logger.info("LoRA enabled: r=%d, alpha=%d", lora_cfg.get("r", 16), lora_cfg.get("alpha", 32))

    # ---- Data ----
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    dataset_name = data_cfg.get("dataset", "gsm8k")
    logger.info("Dataset: %s", dataset_name)

    raw_data = load_gsm8k(
        data_path=data_cfg.get("data_path"),
        split="train",
    )

    # Truncate data if max_samples is specified
    max_samples = data_cfg.get("max_samples", 0)
    if max_samples and max_samples > 0:
        original_len = len(raw_data)
        raw_data = raw_data[:max_samples]
        logger.info("Using max_samples=%d (truncated from %d)", max_samples, original_len)

    curriculum_cfg = cfg.get("curriculum", {})

    # Determine dataset mode: config override > auto-detect
    config_mode = data_cfg.get("mode", None)
    if config_mode:
        dataset_mode = config_mode
    elif num_latent_tokens > 0 and loss_cfg.get("transition_weight", 0) > 0:
        dataset_mode = "student"
    else:
        dataset_mode = "teacher"

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
        mode=dataset_mode,
        num_latent_tokens=num_latent_tokens,
        include_teacher_format=(
            loss_cfg.get("bridge_weight", 0) > 0
            or any(
                s.get("bridge_weight", 0) > 0
                for s in curriculum_cfg.get("stages", [])
            )
        ),
    )

    # Train/Val split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_cfg = cfg.get("training", {})
    pad_token_id = model.tokenizer.pad_token_id or 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Transition Module ----
    transition_module = None
    teacher_states = None

    if loss_cfg.get("transition_weight", 0) > 0:
        layer_ids = list(model_cfg.get("layer_ids", [-1, -2]))
        # Resolve layer ids to positive indices
        num_layers = model.get_num_layers()
        resolved_ids = [
            lid if lid >= 0 else num_layers + 1 + lid for lid in layer_ids
        ]
        transition_module = StateTransitionModule(
            layer_ids=resolved_ids,
        ).to(device)

        # Load pre-cached teacher states
        teacher_cfg = cfg.get("teacher", {})
        cache_dir = teacher_cfg.get("cache_dir", "teacher_states")
        cache_file = os.path.join(cache_dir, "teacher_states.h5")
        if os.path.exists(cache_file):
            from src.data.state_extractor import TeacherStateExtractor

            # Use original (possibly negative) layer_ids to match HDF5 keys
            teacher_states = TeacherStateExtractor.load_cached_states(
                cache_file, layer_ids, device="cpu"  # 保持在CPU，训练时按batch搬到GPU
            )
            logger.info("Loaded teacher states from %s", cache_file)
        else:
            logger.warning(
                "Teacher state cache not found at %s. "
                "Run scripts/extract_teacher_states.py first.",
                cache_file,
            )

    # ---- Curriculum ----
    scheduler = None
    curriculum_cfg = cfg.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        scheduler = CurriculumScheduler(
            stages=list(curriculum_cfg.get("stages", [])),
            enabled=True,
        )
        logger.info("Curriculum scheduler: %s", scheduler)

    # ---- Trainer ----
    trainer_config = {
        "num_epochs": train_cfg.get("num_epochs", 10),
        "learning_rate": train_cfg.get("learning_rate", 5e-5),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 4),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.1),
        "fp16": train_cfg.get("fp16", True),
        "bf16": train_cfg.get("bf16", False),
        "seed": seed,
        # Loss weights
        "transition_weight": loss_cfg.get("transition_weight", 0.7),
        "anchor_weight": loss_cfg.get("anchor_weight", 0.0),
        "bridge_weight": loss_cfg.get("bridge_weight", 0.0),
        "generation_weight": loss_cfg.get("generation_weight", 0.3),
        "normalize_transition": loss_cfg.get("normalize_transition", False),
        "bridge_rho": loss_cfg.get("bridge_rho", 1.0),
        "bridge_xi": loss_cfg.get("bridge_xi", 0.5),
        # Logging
        "use_wandb": cfg.get("logging", {}).get("use_wandb", False),
        "project_name": cfg.get("logging", {}).get("project_name", "latent-reasoning"),
        "run_name": cfg.get("logging", {}).get("run_name"),
        "log_interval": cfg.get("logging", {}).get("log_interval", 50),
        # Checkpoints
        "save_dir": cfg.get("checkpoint", {}).get("save_dir", "checkpoints"),
        "save_interval": cfg.get("checkpoint", {}).get("save_interval", 1),
        "keep_top_k": cfg.get("checkpoint", {}).get("keep_top_k", 15),
        # Eval
        "eval_interval": cfg.get("evaluation", {}).get("eval_interval", 1),
        "eval_metrics": list(cfg.get("evaluation", {}).get("metrics", ["accuracy", "exact_match"])),
        "max_new_tokens": cfg.get("evaluation", {}).get("max_new_tokens", 128),
    }

    trainer = Trainer(
        model=model,
        transition_module=transition_module,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
        teacher_states=teacher_states,
        curriculum_scheduler=scheduler if curriculum_cfg.get("enabled", False) else None,
    )

    # Resume if requested (CLI flag takes priority over config)
    resume_path = args.resume or cfg.get("checkpoint", {}).get("resume_from")
    if resume_path:
        trainer.load_checkpoint(resume_path)

    # ---- Train ----
    trainer.train()

    # ---- Save final model (from best val_loss checkpoint) ----
    final_dir = os.path.join(trainer_config["save_dir"], "final")
    best_ckpt_path = os.path.join(trainer_config["save_dir"], "checkpoint_best.pt")

    if os.path.exists(best_ckpt_path):
        # Load best checkpoint and save as final
        best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        best_epoch = best_state.get("epoch", "?")
        best_val = best_state.get("val_loss", "?")
        logger.info("Loading best checkpoint (epoch=%s, val_loss=%s) for final save", best_epoch, best_val)
    else:
        logger.info("No best checkpoint found, saving last epoch as final")

    model.save_pretrained(final_dir)
    # Also save latent_embeddings separately for easy loading
    latent_path = os.path.join(final_dir, "latent_embeddings.pt")
    torch.save(model.latent_embeddings.state_dict(), latent_path)
    logger.info("Final model saved to %s", final_dir)


if __name__ == "__main__":
    main()
