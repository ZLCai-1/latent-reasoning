"""
Main Trainer for Latent Reasoning.

Native PyTorch training loop with gradient accumulation, mixed-precision,
learning-rate scheduling, WandB logging, and checkpointing.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Full-featured training loop for latent-reasoning models.

    Args:
        model: The :class:`LatentReasoningModel` (student).
        transition_module: :class:`StateTransitionModule` for alignment.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader (optional).
        config: OmegaConf / dict with training hyper-parameters.
        teacher_states: Pre-loaded teacher state cache (dict).
    """

    def __init__(
        self,
        model: nn.Module,
        transition_module: Optional[nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        teacher_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.transition_module = transition_module
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.teacher_states = teacher_states

        # Merge defaults with provided config
        self.cfg = self._default_config()
        if config is not None:
            self.cfg.update(config)

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_metric = float("inf")

        # Set up optimizer, scheduler, scaler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.cfg["fp16"])

        # WandB
        self.wandb_run = None
        if self.cfg.get("use_wandb", False):
            self._init_wandb()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "num_epochs": 10,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "fp16": True,
            "seed": 42,
            # Loss weights
            "transition_weight": 0.7,
            "anchor_weight": 0.0,
            "bridge_weight": 0.0,
            "generation_weight": 0.3,
            "normalize_transition": False,
            # Logging & checkpointing
            "use_wandb": False,
            "project_name": "latent-reasoning",
            "run_name": None,
            "log_interval": 50,
            "save_dir": "checkpoints",
            "save_interval": 1,
            "keep_top_k": 3,
            # Evaluation
            "eval_interval": 1,
        }

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> AdamW:
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.cfg["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        # Include transition module params if available
        if self.transition_module is not None:
            for p in self.transition_module.parameters():
                if p.requires_grad:
                    params[0]["params"].append(p)

        return AdamW(params, lr=self.cfg["learning_rate"])

    def _build_scheduler(self):
        total_steps = (
            len(self.train_dataloader)
            * self.cfg["num_epochs"]
            // self.cfg["gradient_accumulation_steps"]
        )
        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])

        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-8 / max(self.cfg["learning_rate"], 1e-10),
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _init_wandb(self) -> None:
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.cfg.get("project_name", "latent-reasoning"),
                name=self.cfg.get("run_name"),
                config=self.cfg,
                reinit=True,
            )
            logger.info("WandB initialised: %s", self.wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — logging disabled.")
        except Exception as e:
            logger.warning("Failed to init wandb: %s", e)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Starting training for %d epochs", self.cfg["num_epochs"])
        torch.manual_seed(self.cfg["seed"])

        for epoch in range(self.cfg["num_epochs"]):
            train_loss = self._train_epoch(epoch)
            logger.info("Epoch %d — train_loss=%.4f", epoch, train_loss)

            # Validation
            if (
                self.val_dataloader is not None
                and (epoch + 1) % self.cfg["eval_interval"] == 0
            ):
                val_loss = self._validate(epoch)
                logger.info("Epoch %d — val_loss=%.4f", epoch, val_loss)

                if self.wandb_run is not None:
                    import wandb
                    wandb.log({"val_loss": val_loss, "epoch": epoch})

            # Checkpoint
            if (epoch + 1) % self.cfg["save_interval"] == 0:
                self._save_checkpoint(epoch, train_loss)

        logger.info("Training complete.")

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch; returns mean loss."""
        self.model.train()
        if self.transition_module is not None:
            self.transition_module.train()

        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            loss = self._training_step(batch, step)
            total_loss += loss
            num_batches += 1

            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

            # Logging
            if self.global_step % self.cfg["log_interval"] == 0 and self.wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "global_step": self.global_step,
                        "epoch": epoch,
                    }
                )

        return total_loss / max(num_batches, 1)

    def _training_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        """Execute a single training step with gradient accumulation."""
        from ..models.loss_functions import (
            combined_loss,
            generation_loss,
            transition_loss as compute_transition_loss,
            anchor_loss as compute_anchor_loss,
        )

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        boundary_positions = batch.get("boundary_positions")
        if boundary_positions is not None:
            boundary_positions = boundary_positions.to(self.device)

        with autocast(enabled=self.cfg["fp16"]):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Generation loss
            gen_l = generation_loss(outputs["logits"], labels)

            # Transition alignment loss (if applicable)
            trans_l = torch.tensor(0.0, device=self.device)
            anchor_l = torch.tensor(0.0, device=self.device)
            bridge_l = torch.tensor(0.0, device=self.device)

            if (
                self.transition_module is not None
                and boundary_positions is not None
                and self.teacher_states is not None
                and outputs.get("hidden_states") is not None
            ):
                all_hidden = outputs["hidden_states"]  # tuple of tensors

                # Get teacher transitions for this batch
                teacher_trans = self._get_teacher_transitions_for_batch(
                    batch_indices=None,
                    batch_size=input_ids.size(0),
                )

                if teacher_trans is not None:
                    result = self.transition_module(
                        student_hidden_states=all_hidden,
                        boundary_positions=boundary_positions,
                        teacher_transitions=teacher_trans,
                        normalize=self.cfg.get("normalize_transition", False),
                    )
                    trans_l = result["transition_loss"]

            # Combined loss
            weights = {
                "transition": self.cfg["transition_weight"],
                "anchor": self.cfg["anchor_weight"],
                "bridge": self.cfg["bridge_weight"],
                "generation": self.cfg["generation_weight"],
            }
            losses_dict = {
                "transition": trans_l,
                "anchor": anchor_l,
                "bridge": bridge_l,
                "generation": gen_l,
            }
            loss, _loss_info = combined_loss(losses_dict, weights)
            loss = loss / self.cfg["gradient_accumulation_steps"]

        # Backward
        self.scaler.scale(loss).backward()

        # Optimizer step (with gradient accumulation)
        if (step + 1) % self.cfg["gradient_accumulation_steps"] == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["max_grad_norm"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return loss.item() * self.cfg["gradient_accumulation_steps"]

    def _get_teacher_transitions_for_batch(
        self,
        batch_indices: Optional[List[int]],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """Retrieve teacher transitions for the current batch.

        Returns:
            Tensor ``[B, K-1, num_layers, D]`` or ``None``.
        """
        if self.teacher_states is None:
            return None

        transitions_dict = self.teacher_states.get("transitions")
        if transitions_dict is None:
            return None

        # transitions_dict is {layer_id: tensor[N, K-1, D]}
        # Stack all layers into [N, K-1, num_layers, D]
        layer_tensors = list(transitions_dict.values())
        if not layer_tensors:
            return None

        # Each tensor is [N, K-1, D], stack to [N, K-1, num_layers, D]
        transitions = torch.stack(layer_tensors, dim=2)

        start = self.global_step * batch_size
        end = start + batch_size

        if start >= transitions.size(0):
            return None

        actual_end = min(end, transitions.size(0))
        return transitions[start:actual_end].to(self.device)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation and return mean loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        from ..models.loss_functions import generation_loss

        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = generation_loss(outputs["logits"], labels)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save a training checkpoint."""
        save_dir = Path(self.cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.cfg,
        }
        if self.transition_module is not None:
            state["transition_module_state_dict"] = self.transition_module.state_dict()

        torch.save(state, ckpt_path)
        logger.info("Checkpoint saved → %s", ckpt_path)

        # Keep only top-k checkpoints
        self._cleanup_checkpoints(save_dir)

    def _cleanup_checkpoints(self, save_dir: Path) -> None:
        """Remove old checkpoints, keeping only the most recent K."""
        keep_k = self.cfg.get("keep_top_k", 3)
        ckpts = sorted(save_dir.glob("checkpoint_epoch*.pt"), key=os.path.getmtime)
        while len(ckpts) > keep_k:
            oldest = ckpts.pop(0)
            oldest.unlink()
            logger.info("Removed old checkpoint: %s", oldest)

    def load_checkpoint(self, ckpt_path: str) -> int:
        """Load a checkpoint and return the starting epoch.

        Args:
            ckpt_path: Path to the ``.pt`` checkpoint file.

        Returns:
            The epoch number to resume from.
        """
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.global_step = state.get("global_step", 0)

        if (
            self.transition_module is not None
            and "transition_module_state_dict" in state
        ):
            self.transition_module.load_state_dict(
                state["transition_module_state_dict"]
            )

        epoch = state.get("epoch", 0)
        logger.info("Resumed from checkpoint %s (epoch=%d)", ckpt_path, epoch)
        return epoch + 1
