"""
Loss functions for Latent Reasoning training.

Implements four types of losses:
- transition_loss: Align student ΔH with teacher ΔH.
- anchor_loss: Prevent hidden-state drift at boundaries.
- bridge_loss: Mitigate exposure mismatch.
- generation_loss: Standard cross-entropy for answer generation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def transition_loss(
    student_transitions: torch.Tensor,
    teacher_transitions: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Align student state transitions with teacher state transitions.

    Computes MSE between student ΔH and teacher ΔH.  Optionally
    L2-normalizes both before comparison to focus on direction rather
    than magnitude.

    Args:
        student_transitions: ``[B, K-1, ...]`` student ΔH vectors.
        teacher_transitions: ``[B, K-1, ...]`` teacher ΔH vectors.
        normalize: If ``True``, normalize along the last dimension.
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss (mean squared error).
    """
    # Teacher is the target — no gradient.
    teacher_transitions = teacher_transitions.detach()

    # Align K-1 dimension for curriculum learning
    K_s = student_transitions.size(1)
    K_t = teacher_transitions.size(1)
    if K_s < K_t:
        teacher_transitions = teacher_transitions[:, :K_s]
    elif K_t < K_s:
        student_transitions = student_transitions[:, :K_t]

    if normalize:
        # Clamp norms to avoid division-by-zero on zero vectors.
        s_norm = student_transitions.norm(dim=-1, keepdim=True).clamp(min=eps)
        t_norm = teacher_transitions.norm(dim=-1, keepdim=True).clamp(min=eps)
        student_transitions = student_transitions / s_norm
        teacher_transitions = teacher_transitions / t_norm

    return F.mse_loss(student_transitions, teacher_transitions)


def anchor_loss(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Prevent hidden-state drift at boundary positions.

    Encourages the student's absolute hidden states at boundary positions
    to remain close to the teacher's states, preventing accumulating
    drift across multiple transitions.

    Args:
        student_states: ``[B, K, D]`` student boundary states.
        teacher_states: ``[B, K, D]`` teacher boundary states.
        normalize: If ``True``, L2-normalize before comparison.
        eps: Small constant for numerical stability.

    Returns:
        Scalar L2 loss.
    """
    # Align K dimension for curriculum learning
    K_s = student_states.size(1)
    K_t = teacher_states.size(1)
    if K_s < K_t:
        teacher_states = teacher_states[:, :K_s]
    elif K_t < K_s:
        student_states = student_states[:, :K_t]

    teacher_states = teacher_states.detach()

    if normalize:
        s_norm = student_states.norm(dim=-1, keepdim=True).clamp(min=eps)
        t_norm = teacher_states.norm(dim=-1, keepdim=True).clamp(min=eps)
        student_states = student_states / s_norm
        teacher_states = teacher_states / t_norm

    return F.mse_loss(student_states, teacher_states)


def bridge_loss(
    student_states_teacher_prefix: torch.Tensor,
    student_states_self_prefix: torch.Tensor,
    teacher_states: torch.Tensor,
    rho: float = 1.0,
    xi: float = 0.5,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Complete bridge loss (3-term formula) for exposure mismatch.

    L_bridge = Σ d(ŝ_k^{S←T}, stopgrad(s_k^T))          # term1
             + ρ Σ d(ŝ_k^{S←S}, stopgrad(s_k^T))        # term2
             + ξ Σ d(ŝ_k^{S←T}, stopgrad(ŝ_k^{S←S}))   # term3

    Args:
        student_states_teacher_prefix: ``[B, K, ...]`` hidden states from
            student forward with teacher text prefix (ŝ_k^{S←T}).
        student_states_self_prefix: ``[B, K, ...]`` hidden states from
            student forward with its own latent prefix (ŝ_k^{S←S}).
        teacher_states: ``[B, K, ...]`` teacher boundary hidden states
            (s_k^T).
        rho: Weight for term 2 (self-prefix → teacher).
        xi: Weight for term 3 (consistency constraint).
        normalize: If ``True``, L2-normalize all states before comparison.
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss.
    """
    teacher_states = teacher_states.detach()

    # Align K dimension across all three tensors (curriculum may change num_latent)
    K_min = min(
        student_states_teacher_prefix.size(1),
        student_states_self_prefix.size(1),
        teacher_states.size(1),
    )
    student_states_teacher_prefix = student_states_teacher_prefix[:, :K_min]
    student_states_self_prefix = student_states_self_prefix[:, :K_min]
    teacher_states = teacher_states[:, :K_min]

    # Normalize if requested
    if normalize:
        def _norm(x):
            return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
        student_states_teacher_prefix = _norm(student_states_teacher_prefix)
        student_states_self_prefix = _norm(student_states_self_prefix)
        teacher_states = _norm(teacher_states)

    # Term 1: Student(teacher prefix) → Teacher
    term1 = F.mse_loss(student_states_teacher_prefix, teacher_states)

    # Term 2: Student(self prefix) → Teacher
    term2 = rho * F.mse_loss(student_states_self_prefix, teacher_states)

    # Term 3: Consistency — teacher prefix and self prefix outputs should agree
    term3 = xi * F.mse_loss(
        student_states_teacher_prefix, student_states_self_prefix.detach()
    )

    return term1 + term2 + term3


def generation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard cross-entropy loss for next-token prediction.

    Args:
        logits: ``[B, L, V]`` predicted logits.
        labels: ``[B, L]`` target token ids (use *ignore_index* for
                positions that should not contribute to the loss).
        ignore_index: Label value to ignore (default: ``-100``).

    Returns:
        Scalar cross-entropy loss.
    """
    # Shift logits and labels for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


def combined_loss(
    losses_dict: Dict[str, torch.Tensor],
    weights_dict: Optional[Dict[str, float]] = None,
) -> tuple:
    """Weighted combination of all loss components.

    Args:
        losses_dict: Mapping from loss name to its scalar tensor.
            Expected keys: ``"transition"``, ``"anchor"``,
            ``"bridge"``, ``"generation"``.
        weights_dict: Mapping from loss name to scalar weight.
            Defaults to ``transition=0.7, generation=0.3``.

    Returns:
        ``(total_loss, loss_info)`` where *loss_info* is a plain
        ``dict`` with each component's (unweighted) value for logging.
    """
    if weights_dict is None:
        weights_dict = {
            "transition": 0.7,
            "anchor": 0.0,
            "bridge": 0.0,
            "generation": 0.3,
        }

    device = None
    for v in losses_dict.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break

    total = torch.tensor(0.0, device=device)
    loss_info: Dict[str, float] = {}

    for name, loss_val in losses_dict.items():
        w = weights_dict.get(name, 0.0)
        total = total + w * loss_val
        loss_info[name] = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)

    loss_info["total"] = total.item()
    return total, loss_info
