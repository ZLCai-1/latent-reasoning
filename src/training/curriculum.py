"""
Curriculum Learning Scheduler.

Manages a multi-stage training schedule where the number of latent
tokens and loss weights evolve over the course of training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage.

    Attributes:
        num_latent: Number of latent tokens in this stage.
        epochs: How many epochs to train in this stage.
        transition_weight: Weight for transition alignment loss.
        generation_weight: Weight for generation loss.
        anchor_weight: Weight for anchor loss.
        bridge_weight: Weight for bridge loss.
    """

    num_latent: int = 1
    epochs: int = 5
    transition_weight: float = 0.7
    generation_weight: float = 0.3
    anchor_weight: float = 0.0
    bridge_weight: float = 0.0


class CurriculumScheduler:
    """Schedule the number of latent tokens and loss weights over training.

    Supports progressive difficulty by gradually increasing the number
    of latent tokens that replace explicit CoT steps.

    Args:
        stages: List of stage configurations (dicts or
                :class:`CurriculumStage` objects).
        enabled: If ``False``, the scheduler always returns the first
                 stage's config.
    """

    def __init__(
        self,
        stages: Optional[List[Dict[str, Any]]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled

        if stages is None:
            stages = [
                {"num_latent": 1, "epochs": 5, "transition_weight": 0.7},
                {"num_latent": 2, "epochs": 5, "transition_weight": 0.6},
                {"num_latent": 3, "epochs": 5, "transition_weight": 0.5},
            ]

        self.stages: List[CurriculumStage] = []
        for s in stages:
            if isinstance(s, CurriculumStage):
                self.stages.append(s)
            else:
                self.stages.append(CurriculumStage(**s))

        # Pre-compute cumulative epoch boundaries
        self._epoch_boundaries: List[int] = []
        cumulative = 0
        for stage in self.stages:
            cumulative += stage.epochs
            self._epoch_boundaries.append(cumulative)

        self.total_epochs = cumulative
        logger.info(
            "CurriculumScheduler: %d stages, total_epochs=%d, enabled=%s",
            len(self.stages),
            self.total_epochs,
            self.enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_stage(self, epoch: int) -> CurriculumStage:
        """Return the :class:`CurriculumStage` active at *epoch*.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Active curriculum stage.
        """
        if not self.enabled:
            return self.stages[0]

        for i, boundary in enumerate(self._epoch_boundaries):
            if epoch < boundary:
                return self.stages[i]

        # Past the last stage — stick with the final config
        return self.stages[-1]

    def get_current_config(self, global_step: int = 0, epoch: int = 0) -> Dict[str, Any]:
        """Return a configuration dictionary for the current point.

        This is the main interface consumed by the :class:`Trainer`.

        Args:
            global_step: Current global training step (unused for
                         epoch-based scheduling, but reserved for
                         future step-based curricula).
            epoch: Current epoch index (zero-based).

        Returns:
            Dictionary with keys ``num_latent``, ``transition_weight``,
            ``generation_weight``, ``anchor_weight``, ``bridge_weight``.
        """
        stage = self.get_current_stage(epoch)
        return {
            "num_latent": stage.num_latent,
            "transition_weight": stage.transition_weight,
            "generation_weight": stage.generation_weight,
            "anchor_weight": stage.anchor_weight,
            "bridge_weight": stage.bridge_weight,
        }

    def get_stage_index(self, epoch: int) -> int:
        """Return the 0-based index of the stage at *epoch*."""
        if not self.enabled:
            return 0
        for i, boundary in enumerate(self._epoch_boundaries):
            if epoch < boundary:
                return i
        return len(self.stages) - 1

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        lines = [f"CurriculumScheduler(enabled={self.enabled}, stages=["]
        for i, s in enumerate(self.stages):
            lines.append(f"  Stage {i}: {s}")
        lines.append("])")
        return "\n".join(lines)
