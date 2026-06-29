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
        curriculum_scheduler=None,
    ) -> None:
        self.model = model
        self.transition_module = transition_module
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.teacher_states = teacher_states
        self.curriculum_scheduler = curriculum_scheduler

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

        # Separate latent_embeddings for higher learning rate
        latent_params = []
        other_params_decay = []
        other_params_no_decay = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "latent_embeddings" in n:
                latent_params.append(p)
            elif any(nd in n for nd in no_decay):
                other_params_no_decay.append(p)
            else:
                other_params_decay.append(p)

        params = [
            {"params": other_params_decay, "weight_decay": self.cfg["weight_decay"]},
            {"params": other_params_no_decay, "weight_decay": 0.0},
            {"params": latent_params, "lr": self.cfg["learning_rate"] * 100, "weight_decay": 0.0},
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
            # Update loss weights from curriculum if available
            if self.curriculum_scheduler is not None:
                stage = self.curriculum_scheduler.get_current_stage(epoch)
                if stage is not None:
                    self.cfg["transition_weight"] = stage.transition_weight
                    self.cfg["generation_weight"] = stage.generation_weight
                    self.cfg["anchor_weight"] = getattr(stage, 'anchor_weight', 0.0)
                    self.cfg["bridge_weight"] = getattr(stage, 'bridge_weight', 0.0)

            train_loss, loss_breakdown = self._train_epoch(epoch)
            # Format loss breakdown for logging
            parts = [f"train_loss={train_loss:.4f}"]
            for k in ["transition", "anchor", "bridge", "generation"]:
                if k in loss_breakdown and loss_breakdown[k] > 0:
                    parts.append(f"{k}={loss_breakdown[k]:.4f}")
            logger.info("Epoch %d \u2014 %s", epoch, ", ".join(parts))

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

    def _train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch; returns (mean_loss, loss_breakdown)."""
        self.model.train()
        if self.transition_module is not None:
            self.transition_module.train()

        total_loss = 0.0
        total_breakdown = {}
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            loss, loss_info = self._training_step(batch, step)
            total_loss += loss
            num_batches += 1

            # Accumulate loss breakdown
            for k, v in loss_info.items():
                total_breakdown[k] = total_breakdown.get(k, 0.0) + v

            # Build postfix with individual loss components
            postfix = {"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
            if loss_info.get("anchor", 0) > 0:
                postfix["anc"] = f"{loss_info['anchor']:.4f}"
            if loss_info.get("bridge", 0) > 0:
                postfix["brg"] = f"{loss_info['bridge']:.4f}"
            pbar.set_postfix(**postfix)

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

        avg_loss = total_loss / max(num_batches, 1)
        avg_breakdown = {k: v / max(num_batches, 1) for k, v in total_breakdown.items()}
        return avg_loss, avg_breakdown

    def _training_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        """Execute a single training step with gradient accumulation.

        Supports two modes:
        - **Student mode** (when batch contains ``latent_positions``):
          Uses ``model.forward_with_latent`` to inject learned latent
          embeddings, then aligns hidden states at latent positions
          with pre-cached teacher transitions.
        - **Teacher mode** (legacy, when batch contains
          ``boundary_positions``): Full CoT forward with boundary
          alignment.
        """
        from ..models.loss_functions import (
            combined_loss,
            generation_loss,
            transition_loss as compute_transition_loss,
            anchor_loss as compute_anchor_loss,
            bridge_loss as compute_bridge_loss,
        )

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Detect mode based on batch contents
        latent_positions = batch.get("latent_positions")
        boundary_positions = batch.get("boundary_positions")
        is_student_mode = latent_positions is not None

        if latent_positions is not None:
            latent_positions = latent_positions.to(self.device)
        if boundary_positions is not None:
            boundary_positions = boundary_positions.to(self.device)

        with autocast(enabled=self.cfg["fp16"]):
            if is_student_mode:
                # --- Student forward pass ---
                outputs = self.model.forward_with_latent(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    latent_positions=latent_positions,
                    output_hidden_states=True,
                )
            else:
                # --- Teacher/legacy forward pass ---
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Generation loss (answer portion)
            gen_l = generation_loss(outputs["logits"], labels)

            # Transition alignment loss
            trans_l = torch.tensor(0.0, device=self.device)
            anchor_l = torch.tensor(0.0, device=self.device)
            bridge_l = torch.tensor(0.0, device=self.device)
            student_boundary = None
            teacher_boundary = None

            if (
                self.transition_module is not None
                and self.teacher_states is not None
                and outputs.get("hidden_states") is not None
            ):
                all_hidden = outputs["hidden_states"]  # tuple of tensors

                # Get teacher transitions for this batch
                sample_indices = batch.get("sample_idx")
                batch_idx_list = sample_indices.tolist() if sample_indices is not None else None
                teacher_trans = self._get_teacher_transitions_for_batch(
                    batch_indices=batch_idx_list,
                    batch_size=input_ids.size(0),
                )

                if teacher_trans is not None:
                    if is_student_mode:
                        # In student mode, use latent_positions as
                        # boundary positions for state extraction.
                        # latent_positions [B, K] → the K positions
                        # where hidden states should match teacher.
                        result = self.transition_module(
                            student_hidden_states=all_hidden,
                            boundary_positions=latent_positions,
                            teacher_transitions=teacher_trans,
                            normalize=self.cfg.get("normalize_transition", False),
                        )
                    elif boundary_positions is not None:
                        result = self.transition_module(
                            student_hidden_states=all_hidden,
                            boundary_positions=boundary_positions,
                            teacher_transitions=teacher_trans,
                            normalize=self.cfg.get("normalize_transition", False),
                        )
                    else:
                        result = None

                    if result is not None:
                        trans_l = result["transition_loss"]
                        student_boundary = result["student_boundary_states"]

                # --- Anchor loss ---
                if (
                    self.cfg["anchor_weight"] > 0
                    and student_boundary is not None
                ):
                    teacher_boundary = self._get_teacher_boundary_states_for_batch(
                        input_ids.size(0), batch_indices=batch_idx_list
                    )
                    if teacher_boundary is not None:
                        anchor_l = compute_anchor_loss(
                            student_boundary, teacher_boundary,
                            normalize=self.cfg.get("normalize_transition", False),
                        )

                # --- Bridge loss (3-term) ---
                if (
                    self.cfg["bridge_weight"] > 0
                    and student_boundary is not None
                    and is_student_mode
                ):
                    if teacher_boundary is None:
                        teacher_boundary = (
                            self._get_teacher_boundary_states_for_batch(
                                input_ids.size(0), batch_indices=batch_idx_list
                            )
                        )
                    s_teacher_prefix = self._bridge_forward_teacher_prefix(
                        batch
                    )
                    if (
                        s_teacher_prefix is not None
                        and teacher_boundary is not None
                    ):
                        bridge_l = compute_bridge_loss(
                            student_states_teacher_prefix=s_teacher_prefix,
                            student_states_self_prefix=student_boundary,
                            teacher_states=teacher_boundary,
                            rho=self.cfg.get("bridge_rho", 1.0),
                            xi=self.cfg.get("bridge_xi", 0.5),
                            normalize=self.cfg.get("normalize_transition", False),
                        )

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
            loss, loss_info = combined_loss(losses_dict, weights)
            loss = loss / self.cfg["gradient_accumulation_steps"]

        # Backward
        self.scaler.scale(loss).backward()

        # Optimizer step (with gradient accumulation)
        if (step + 1) % self.cfg["gradient_accumulation_steps"] == 0:
            self.scaler.unscale_(self.optimizer)
            # Clip gradients for both model and latent embeddings
            all_params = list(self.model.parameters())
            if self.transition_module is not None:
                all_params += list(self.transition_module.parameters())
            nn.utils.clip_grad_norm_(all_params, self.cfg["max_grad_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return loss.item() * self.cfg["gradient_accumulation_steps"], loss_info

    def _get_teacher_transitions_for_batch(
        self,
        batch_indices: Optional[List[int]],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """Retrieve teacher transitions for the current batch.

        Uses sample_idx from batch for precise alignment when available,
        otherwise falls back to sequential indexing.

        Returns:
            Tensor ``[B, K-1, num_layers, D]`` or ``None``.
        """
        if self.teacher_states is None:
            return None

        # Prefer computing transitions from boundary_states (SPAN_END only)
        boundary_dict = self.teacher_states.get("boundary_states")
        if boundary_dict and len(boundary_dict) > 0:
            layer_tensors = list(boundary_dict.values())
            if layer_tensors:
                # Stack: [N, K_all, num_layers, D]
                boundary = torch.stack(layer_tensors, dim=2)
                # Take only SPAN_END positions (odd indices)
                boundary_end = boundary[:, 1::2, :, :]  # [N, K, nL, D]
                # Compute transitions between consecutive SPAN_END states
                from ..models.state_transition import StateTransitionModule
                transitions = StateTransitionModule.compute_transitions(
                    boundary_end
                )  # [N, K-1, nL, D]

                # Use sample_idx for precise alignment
                if batch_indices is not None:
                    indices = [i for i in batch_indices if i < transitions.size(0)]
                    if indices:
                        return transitions[indices].to(self.device)
                    return None

                # Fallback: sequential (legacy)
                start = self.global_step * batch_size
                end = start + batch_size
                if start >= transitions.size(0):
                    start = start % transitions.size(0)
                    end = start + batch_size
                actual_end = min(end, transitions.size(0))
                return transitions[start:actual_end].to(self.device)

        # Fallback: use raw cached transitions
        transitions_dict = self.teacher_states.get("transitions")
        if transitions_dict is None or len(transitions_dict) == 0:
            return None

        layer_tensors = list(transitions_dict.values())
        if not layer_tensors:
            return None

        transitions = torch.stack(layer_tensors, dim=2)

        # Use sample_idx for precise alignment
        if batch_indices is not None:
            indices = [i for i in batch_indices if i < transitions.size(0)]
            if indices:
                return transitions[indices].to(self.device)
            return None

        # Fallback: sequential (legacy)
        start = self.global_step * batch_size
        end = start + batch_size
        if start >= transitions.size(0):
            start = start % transitions.size(0)
            end = start + batch_size
        actual_end = min(end, transitions.size(0))
        return transitions[start:actual_end].to(self.device)

    def _get_teacher_boundary_states_for_batch(
        self,
        batch_size: int,
        batch_indices: Optional[List[int]] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve teacher boundary states (SPAN_END only) for the batch.

        Returns:
            Tensor ``[B, K, num_layers, D]`` or ``None``.
            K = num_spans (only SPAN_END positions are retained).
        """
        if self.teacher_states is None:
            return None

        boundary_dict = self.teacher_states.get("boundary_states")
        if boundary_dict is None or len(boundary_dict) == 0:
            return None

        # boundary_dict is {layer_id: tensor[N, K_all, D]}
        # K_all = 2 * num_spans (SPAN_START and SPAN_END alternating)
        layer_tensors = list(boundary_dict.values())
        if not layer_tensors:
            return None

        # Stack to [N, K_all, num_layers, D]
        boundary = torch.stack(layer_tensors, dim=2)

        # Take only SPAN_END positions (odd indices: 1, 3, 5, ...)
        boundary = boundary[:, 1::2, :, :]  # [N, K, num_layers, D]

        # Use sample_idx for precise alignment
        if batch_indices is not None:
            indices = [i for i in batch_indices if i < boundary.size(0)]
            if indices:
                return boundary[indices].to(self.device)
            return None

        # Fallback: sequential (legacy)
        start = self.global_step * batch_size
        end = start + batch_size

        if start >= boundary.size(0):
            start = start % boundary.size(0)
            end = start + batch_size

        actual_end = min(end, boundary.size(0))
        return boundary[start:actual_end].to(self.device)

    def _bridge_forward_teacher_prefix(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """S←T forward: inject student latent embeddings into teacher text.

        Runs a forward pass using the teacher-format input_ids (full CoT)
        but replaces embeddings at SPAN_END positions with the student's
        learned latent embeddings.

        Returns:
            Tensor ``[B, K, num_layers, D]`` hidden states at injection
            positions, or ``None`` if teacher format data is unavailable.
        """
        teacher_input_ids = batch.get("teacher_input_ids")
        teacher_attention_mask = batch.get("teacher_attention_mask")
        teacher_boundary_positions = batch.get("teacher_boundary_positions")

        if teacher_input_ids is None or teacher_boundary_positions is None:
            return None

        teacher_input_ids = teacher_input_ids.to(self.device)
        teacher_attention_mask = teacher_attention_mask.to(self.device)
        teacher_boundary_positions = teacher_boundary_positions.to(self.device)

        # Get base token embeddings for teacher input
        inputs_embeds = self.model.model.get_input_embeddings()(teacher_input_ids).clone()

        # SPAN_END positions are at odd indices of boundary_positions
        span_end_positions = teacher_boundary_positions[:, 1::2]  # [B, K]
        B, K = span_end_positions.shape

        # Inject student latent embeddings at SPAN_END positions
        num_latent = min(K, self.model.num_latent_tokens)
        for k in range(num_latent):
            latent_emb = self.model.latent_embeddings(
                torch.tensor(k, device=self.device)
            )  # [D]
            for b in range(B):
                pos = span_end_positions[b, k].item()
                if pos > 0:  # skip padding
                    inputs_embeds[b, pos, :] = latent_emb

        # Forward pass with modified embeddings
        outputs = self.model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            output_hidden_states=True,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs["hidden_states"]

        # Extract hidden states at SPAN_END positions
        from ..models.state_transition import StateTransitionModule

        # Use only the first num_latent SPAN_END positions
        extraction_positions = span_end_positions[:, :num_latent]
        boundary_states = StateTransitionModule.extract_boundary_states(
            hidden_states, extraction_positions, self.transition_module.layer_ids
        )
        return boundary_states  # [B, K, num_layers, D]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation and return mean loss.

        Uses forward_with_latent() when latent_positions is present
        (student mode) to ensure validation loss accurately reflects
        the student model's performance with learned latent embeddings.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        from ..models.loss_functions import generation_loss

        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            latent_positions = batch.get("latent_positions")

            if latent_positions is not None:
                latent_positions = latent_positions.to(self.device)
                outputs = self.model.forward_with_latent(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    latent_positions=latent_positions,
                )
            else:
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
