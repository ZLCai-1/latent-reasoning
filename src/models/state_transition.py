"""
State Transition Module.

Core module for computing and aligning hidden-state transitions (ΔH)
between teacher and student models at span boundary positions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StateTransitionModule(nn.Module):
    """Compute and align state transitions between boundary positions.

    Given hidden states from a student model and pre-extracted teacher
    transitions, this module:

    1. Extracts hidden states at span-boundary positions.
    2. Computes transition vectors  ΔH_k = H_k − H_{k-1}.
    3. Measures alignment between student and teacher transitions.

    Args:
        layer_ids: Resolved (positive) layer indices to align.
        hidden_dim: Dimensionality of hidden states (used only when
                    *projection* is ``True``).
        projection: If ``True``, learn a linear projection from student
                    hidden space to teacher hidden space.
    """

    def __init__(
        self,
        layer_ids: List[int],
        hidden_dim: int = 768,
        projection: bool = False,
    ) -> None:
        super().__init__()
        self.layer_ids = layer_ids

        # Optional learnable projection  student_dim → teacher_dim
        self.projectors: Optional[nn.ModuleDict] = None
        if projection:
            self.projectors = nn.ModuleDict(
                {
                    str(lid): nn.Linear(hidden_dim, hidden_dim, bias=False)
                    for lid in layer_ids
                }
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def extract_boundary_states(
        hidden_states: Tuple[torch.Tensor, ...],
        boundary_positions: torch.Tensor,
        layer_ids: List[int],
    ) -> torch.Tensor:
        """Extract hidden vectors at boundary positions for specified layers.

        Args:
            hidden_states: Tuple of ``(num_layers,)`` tensors, each
                ``[B, L, D]`` — as returned by
                ``model(output_hidden_states=True).hidden_states``.
            boundary_positions: Long tensor ``[B, K]`` with boundary
                token positions per sample.
            layer_ids: Which layers to extract (supports negative
                indices such as ``-1``, ``-2``).

        Returns:
            Tensor of shape ``[B, K, num_layers, D]``.
        """
        num_total = len(hidden_states)
        collected = []
        for lid in layer_ids:
            resolved = lid if lid >= 0 else num_total + lid
            if not (0 <= resolved < num_total):
                raise IndexError(
                    f"Layer index {lid} (resolved={resolved}) "
                    f"out of range [0, {num_total})"
                )
            hs = hidden_states[resolved]  # [B, L, D]
            B, _L, D = hs.shape
            K = boundary_positions.shape[1]
            idx = boundary_positions.unsqueeze(-1).expand(B, K, D)  # [B, K, D]
            collected.append(torch.gather(hs, dim=1, index=idx))  # [B, K, D]
        # Stack along a new layer dimension → [B, K, num_layers, D]
        return torch.stack(collected, dim=2)

    @staticmethod
    def compute_transitions(
        boundary_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute transition vectors between consecutive boundary states.

        Args:
            boundary_states: Tensor of shape ``[B, K, ...]`` (e.g.
                ``[B, K, num_layers, D]``) representing hidden states
                at ``K`` consecutive boundary positions.

        Returns:
            Transition deltas ``[B, K-1, ...]``:
            ``ΔH_k = H_k − H_{k-1}`` for ``k = 1 … K-1``.
        """
        return boundary_states[:, 1:] - boundary_states[:, :-1]

    def forward(
        self,
        student_hidden_states: Tuple[torch.Tensor, ...],
        boundary_positions: torch.Tensor,
        teacher_transitions: torch.Tensor,
        normalize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute transition alignment between student and teacher.

        Args:
            student_hidden_states: Tuple of layer tensors from
                ``model(output_hidden_states=True).hidden_states``.
            boundary_positions: ``[B, K]`` positions of span boundaries.
            teacher_transitions: ``[B, K-1, num_layers, D]``
                pre-computed teacher transition vectors.
            normalize: If ``True``, L2-normalize transitions before
                computing the distance.

        Returns:
            Dictionary with:
              - ``"transition_loss"``: scalar mean L2 loss across layers.
              - ``"student_transitions"``: ``[B, K-1, num_layers, D]``.
              - ``"student_boundary_states"``: ``[B, K, num_layers, D]``.
        """
        from .loss_functions import transition_loss as _transition_loss

        # Step 1: Extract boundary states  [B, K, num_layers, D]
        student_boundary = self.extract_boundary_states(
            student_hidden_states, boundary_positions, self.layer_ids,
        )

        # Optional per-layer projection
        if self.projectors is not None:
            projected = []
            for i, lid in enumerate(self.layer_ids):
                s = student_boundary[:, :, i, :]          # [B, K, D]
                key = str(lid)
                if key in self.projectors:
                    s = self.projectors[key](s)
                projected.append(s)
            student_boundary = torch.stack(projected, dim=2)  # [B, K, nL, D]

        # Step 2: Compute student transitions  [B, K-1, num_layers, D]
        s_trans = self.compute_transitions(student_boundary)

        # Step 3: Alignment loss (teacher detached inside transition_loss)
        loss = _transition_loss(s_trans, teacher_transitions, normalize=normalize)

        return {
            "transition_loss": loss,
            "student_transitions": s_trans,
            "student_boundary_states": student_boundary,
        }
