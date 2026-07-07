#!/usr/bin/env python3
"""
扫描训练日志，找到 val_loss 最低的 epoch，将对应 checkpoint 导出为 final/ 目录。

Usage:
    # 自动找 best epoch 并导出
    python scripts/select_best_checkpoint.py \
        --ckpt_dir checkpoints/ablation/no_transition \
        --base_model models/gpt2-local

    # 指定具体 epoch
    python scripts/select_best_checkpoint.py \
        --ckpt_dir checkpoints/ablation/no_transition \
        --base_model models/qwen2.5-math-1.5b \
        --epoch 6

    # 批量处理所有消融
    python scripts/select_best_checkpoint.py \
        --batch_root checkpoints/ablation \
        --base_model models/gpt2-local
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_train_log(log_path):
    """从 train.log 解析每个 epoch 的 val_loss，返回 {epoch: val_loss}."""
    val_losses = {}
    if not os.path.exists(log_path):
        return val_losses
    pattern = re.compile(r"Epoch (\d+) .* val_loss=([\d.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                val_losses[int(m.group(1))] = float(m.group(2))
    return val_losses


def find_best_epoch(ckpt_dir):
    """扫描 train.log 找 val_loss 最低且 checkpoint 还存在的 epoch."""
    log_path = os.path.join(ckpt_dir, "train.log")
    val_losses = parse_train_log(log_path)
    if not val_losses:
        print(f"[WARN] No val_loss found in {log_path}")
        return None

    # 检查哪些 checkpoint 还存在
    available = {}
    for epoch, vl in val_losses.items():
        ckpt_file = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
        if os.path.exists(ckpt_file):
            available[epoch] = vl

    if not available:
        print(f"[WARN] No checkpoint files exist in {ckpt_dir}")
        print(f"[INFO] val_losses recorded: {val_losses}")
        return None

    best_epoch = min(available, key=lambda e: available[e])
    print(f"[INFO] {ckpt_dir}: best epoch = {best_epoch} (val_loss={available[best_epoch]:.4f})")
    print(f"[INFO]   available epochs: {sorted(available.keys())}")
    return best_epoch


def export_final(ckpt_dir, base_model, epoch, num_latent_tokens=3, layer_ids=(-1, -2),
                 use_lora=True, lora_r=128, lora_alpha=32,
                 lora_targets=("c_attn", "c_proj"),
                 lora_dropout=0.0):
    """加载指定 epoch 的 checkpoint，导出到 final/ 目录."""
    import torch
    from src.models.base import LatentReasoningModel

    ckpt_file = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
    final_dir = os.path.join(ckpt_dir, "final")

    if not os.path.exists(ckpt_file):
        print(f"[ERROR] Checkpoint not found: {ckpt_file}")
        return False

    print(f"[INFO] Loading {ckpt_file} ...")
    model = LatentReasoningModel(
        model_name=base_model,
        layer_ids=list(layer_ids),
        num_latent_tokens=num_latent_tokens,
        device="cpu",
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
        )
        model.model = get_peft_model(model.model, lora_config)

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    os.makedirs(final_dir, exist_ok=True)
    model.model.save_pretrained(final_dir)
    model.tokenizer.save_pretrained(final_dir)
    torch.save(model.latent_embeddings.state_dict(),
               os.path.join(final_dir, "latent_embeddings.pt"))
    print(f"[OK] Exported -> {final_dir} (from epoch {epoch})")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Select best checkpoint and export to final/")
    p.add_argument("--ckpt_dir", type=str, help="Single checkpoint directory")
    p.add_argument("--batch_root", type=str, help="Root containing multiple ablation dirs")
    p.add_argument("--epoch", type=int, default=None,
                   help="Specific epoch (default: auto-pick best val_loss)")
    p.add_argument("--base_model", type=str, default="models/gpt2-local")
    p.add_argument("--num_latent_tokens", type=int, default=3)
    p.add_argument("--layer_ids", type=int, nargs="+", default=[-1, -2])
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_targets", type=str, nargs="+", default=["c_attn", "c_proj"])
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_lora", action="store_true", help="Skip LoRA wrapping")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.ckpt_dir and not args.batch_root:
        print("ERROR: Specify either --ckpt_dir or --batch_root")
        sys.exit(1)

    targets = []
    if args.ckpt_dir:
        targets.append(args.ckpt_dir)
    if args.batch_root:
        for d in sorted(glob.glob(os.path.join(args.batch_root, "*"))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "train.log")):
                targets.append(d)

    print(f"[INFO] Processing {len(targets)} checkpoint directories")
    success, failed = 0, 0
    for ckpt_dir in targets:
        print(f"\n=== {ckpt_dir} ===")
        epoch = args.epoch if args.epoch is not None else find_best_epoch(ckpt_dir)
        if epoch is None:
            failed += 1
            continue
        try:
            ok = export_final(
                ckpt_dir=ckpt_dir,
                base_model=args.base_model,
                epoch=epoch,
                num_latent_tokens=args.num_latent_tokens,
                layer_ids=args.layer_ids,
                use_lora=not args.no_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_targets=tuple(args.lora_targets),
                lora_dropout=args.lora_dropout,
            )
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {ckpt_dir}: {e}")
            failed += 1

    print(f"\n========================================")
    print(f"  Success: {success} / Failed: {failed}")
    print(f"========================================")


if __name__ == "__main__":
    main()
