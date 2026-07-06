"""
Teacher Hidden-State Extractor.

Runs a frozen teacher model on CoT-annotated data and caches the
hidden states at boundary positions to HDF5 files for efficient
reuse during student training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TeacherStateExtractor:
    """Extract and cache teacher hidden states at span boundaries.

    This class loads a teacher model, processes the dataset in batches,
    extracts hidden states at the boundary token positions for specified
    layers, and stores them in HDF5 format (fp16) for efficient I/O.

    Args:
        model_name: HuggingFace model identifier or path to a trained
                    checkpoint.
        layer_ids: Layer indices for hidden-state extraction (negative
                   values are resolved from the end).
        device: Target device.
        cache_dir: Directory for HDF5 cache files.
        store_fp16: Whether to store states in float16 to save space.
    """

    def __init__(
        self,
        model_name: str,
        layer_ids: List[int],
        device: str = "cuda",
        cache_dir: str = "teacher_states",
        store_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.layer_ids = layer_ids
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.store_fp16 = store_fp16

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy model loading
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazily load the teacher model."""
        if self._model is None:
            from .dataset import LatentReasoningDataset  # noqa: avoid circular
            from ..models.base import LatentReasoningModel

            logger.info("Loading teacher model: %s", self.model_name)
            wrapper = LatentReasoningModel(
                model_name=self.model_name,
                layer_ids=self.layer_ids,
                device=self.device,
            )
            wrapper.freeze()
            self._model = wrapper
            self._tokenizer = wrapper.tokenizer
        return self._model

    @property
    def tokenizer(self):
        """Access tokenizer (triggers model loading if needed)."""
        _ = self.model  # ensure loaded
        return self._tokenizer

    def extract_and_cache(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        cache_filename: str = "teacher_states.h5",
        num_workers: int = 0,
    ) -> Path:
        """Run extraction on the full dataset and save to HDF5.

        Args:
            dataset: PyTorch dataset returning dicts with ``input_ids``,
                     ``attention_mask``, and ``boundary_positions``.
            batch_size: Batch size for forward passes.
            cache_filename: Name of the output HDF5 file.
            num_workers: DataLoader workers.

        Returns:
            Path to the saved HDF5 file.
        """
        output_path = self.cache_dir / cache_filename
        if output_path.exists():
            logger.info("Cache already exists at %s — skipping extraction.", output_path)
            return output_path

        from functools import partial
        from src.data.dataset import collate_fn as _collate_fn

        pad_token_id = self.model.tokenizer.pad_token_id or 0

        logger.info("Starting teacher state extraction → %s", output_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(_collate_fn, pad_token_id=pad_token_id),
            drop_last=False,
        )

        all_transitions: Dict[int, List[np.ndarray]] = {lid: [] for lid in self.layer_ids}
        all_boundary_states: Dict[int, List[np.ndarray]] = {lid: [] for lid in self.layer_ids}

        model = self.model
        model.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting teacher states"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                boundary_positions = batch.get("boundary_positions")

                if boundary_positions is None:
                    logger.warning("Batch without boundary_positions — skipping.")
                    continue

                boundary_positions = boundary_positions.to(self.device)

                # Forward pass to get hidden states (raw tuple)
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states  # tuple of (num_layers+1,) tensors

                # Extract boundary states and compute transitions
                from ..models.state_transition import StateTransitionModule

                boundary_states = StateTransitionModule.extract_boundary_states(
                    hidden_states, boundary_positions, self.layer_ids
                )  # [B, K, num_layers, D]

                transitions = StateTransitionModule.compute_transitions(
                    boundary_states
                )  # [B, K-1, num_layers, D]

                # Store per layer
                for i, lid in enumerate(self.layer_ids):
                    states_layer = boundary_states[:, :, i, :]  # [B, K, D]
                    trans_layer = transitions[:, :, i, :]  # [B, K-1, D]

                    states_np = states_layer.cpu().float().numpy()
                    trans_np = trans_layer.cpu().float().numpy()

                    if self.store_fp16:
                        states_np = states_np.astype(np.float16)
                        trans_np = trans_np.astype(np.float16)

                    all_boundary_states[lid].append(states_np)
                    all_transitions[lid].append(trans_np)

        # Save to HDF5
        self._save_hdf5(output_path, all_transitions, all_boundary_states)
        logger.info("Teacher states saved to %s", output_path)
        return output_path

    def _resolve_layer_id(
        self, lid: int, hidden_states_dict: Dict[int, torch.Tensor]
    ) -> int:
        """Resolve a possibly-negative layer id to its positive key."""
        if lid in hidden_states_dict:
            return lid
        # The model wrapper already resolves to positive ids
        for key in hidden_states_dict:
            return key  # fallback: return first available key
        return lid

    def _save_hdf5(
        self,
        path: Path,
        transitions: Dict[int, List[np.ndarray]],
        boundary_states: Dict[int, List[np.ndarray]],
    ) -> None:
        """Save extracted states to an HDF5 file."""
        with h5py.File(path, "w") as f:
            for lid in transitions:
                if transitions[lid]:
                    trans_arr = np.concatenate(transitions[lid], axis=0)
                    f.create_dataset(
                        f"transitions/layer_{lid}",
                        data=trans_arr,
                        compression="gzip",
                        compression_opts=4,
                    )
                if boundary_states[lid]:
                    states_arr = np.concatenate(boundary_states[lid], axis=0)
                    f.create_dataset(
                        f"boundary_states/layer_{lid}",
                        data=states_arr,
                        compression="gzip",
                        compression_opts=4,
                    )

            # Store metadata
            f.attrs["layer_ids"] = self.layer_ids
            f.attrs["model_name"] = self.model_name
            f.attrs["store_fp16"] = self.store_fp16

    @staticmethod
    def load_cached_states(
        cache_path: str,
        layer_ids: List[int],
        device: str = "cpu",
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """Load pre-cached teacher states from HDF5.

        Args:
            cache_path: Path to the HDF5 file.
            layer_ids: Layers to load.
            device: Target device for tensors.

        Returns:
            Dictionary with ``"transitions"`` and ``"boundary_states"``
            keys, each mapping layer ids to tensors.
        """
        result: Dict[str, Dict[int, torch.Tensor]] = {
            "transitions": {},
            "boundary_states": {},
        }

        with h5py.File(cache_path, "r") as f:
            for lid in layer_ids:
                trans_key = f"transitions/layer_{lid}"
                if trans_key in f:
                    arr = f[trans_key][:]
                    result["transitions"][lid] = torch.from_numpy(
                        arr
                    ).half().to(device)

                states_key = f"boundary_states/layer_{lid}"
                if states_key in f:
                    arr = f[states_key][:]
                    result["boundary_states"][lid] = torch.from_numpy(
                        arr
                    ).half().to(device)

        logger.info(
            "Loaded cached states from %s (layers=%s)", cache_path, layer_ids
        )
        return result
