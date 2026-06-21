"""
Dataset module for Latent Reasoning.

Handles loading and tokenizing math reasoning datasets (primarily GSM8K)
with support for span-based boundary markers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_gsm8k(
    data_path: Optional[str] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load the GSM8K dataset.

    Supports loading from:
    1. A local JSON / JSONL file (if *data_path* is provided).
    2. HuggingFace ``datasets`` library (default).

    Each returned record has keys:
    - ``question`` (str)
    - ``answer`` (str): final numerical answer
    - ``steps`` (list[str]): intermediate CoT reasoning steps

    Args:
        data_path: Path to a local JSON/JSONL file.  If ``None``,
                   downloads from HuggingFace.
        split: Dataset split (``"train"`` / ``"test"``).

    Returns:
        List of data dictionaries.
    """
    if data_path is not None and os.path.exists(data_path):
        return _load_local(data_path)
    return _load_from_hf(split)


def _load_local(path: str) -> List[Dict[str, Any]]:
    """Load from a local JSON or JSONL file."""
    path = Path(path)
    records: List[Dict[str, Any]] = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")

    # Normalise keys
    normalised: List[Dict[str, Any]] = []
    for rec in records:
        normalised.append(_normalise_record(rec))
    logger.info("Loaded %d records from %s", len(normalised), path)
    return normalised


def _load_from_hf(split: str) -> List[Dict[str, Any]]:
    """Load GSM8K from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required to download GSM8K. "
            "Install it with: pip install datasets"
        )

    logger.info("Downloading GSM8K split=%s from HuggingFace …", split)
    ds = load_dataset("openai/gsm8k", "main", split=split)

    records: List[Dict[str, Any]] = []
    for item in ds:
        records.append(_normalise_record(item))
    logger.info("Loaded %d records from HuggingFace (split=%s)", len(records), split)
    return records


def _normalise_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw GSM8K record into the canonical format."""
    question = rec.get("question", "")
    raw_answer = rec.get("answer", "")

    # GSM8K format: reasoning lines separated by '\n', last line has '#### <number>'
    if isinstance(raw_answer, str) and "####" in raw_answer:
        parts = raw_answer.split("####")
        reasoning = parts[0].strip()
        final_answer = parts[1].strip() if len(parts) > 1 else ""
        steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    else:
        steps = rec.get("steps", [])
        final_answer = str(raw_answer)

    result = {
        "question": question,
        "answer": final_answer,
        "steps": steps,
    }

    # Preserve pre-computed spans if present
    if "spans" in rec:
        result["spans"] = rec["spans"]

    return result


class LatentReasoningDataset(Dataset):
    """PyTorch dataset for latent-reasoning training.

    Tokenizes question + CoT spans + answer with boundary markers
    and returns tensors ready for the model.

    Args:
        data: List of data dictionaries (from :func:`load_gsm8k`).
        tokenizer: HuggingFace tokenizer (with special tokens added).
        max_seq_length: Maximum sequence length after tokenization.
        num_spans: Number of CoT spans (``K``).
        span_strategy: How to partition steps into spans
                       (``"fixed"`` / ``"random"`` / ``"none"``).
        teacher_states_dir: Optional path to HDF5 cache with teacher
                            hidden states.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 512,
        num_spans: int = 3,
        span_strategy: str = "fixed",
        teacher_states_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_spans = num_spans
        self.span_strategy = span_strategy
        self.teacher_states_dir = teacher_states_dir

        # Lazy import to avoid circular dependency
        from .preprocessing import prepare_training_sample

        self._prepare_fn = prepare_training_sample

        # Pre-tokenize if span_strategy is not "none"
        logger.info(
            "Dataset created: %d samples, num_spans=%d, strategy=%s",
            len(data),
            num_spans,
            span_strategy,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.data[idx]

        # Require pre-computed spans
        if "spans" not in record:
            raise ValueError(
                f"Record at index {idx} is missing the 'spans' field. "
                "Please run 'python scripts/preprocess_data.py' first to "
                "generate pre-computed spans."
            )
        spans = record["spans"]

        sample = self._prepare_fn(
            question=record["question"],
            spans=spans,
            answer=record["answer"],
            tokenizer=self.tokenizer,
            num_spans=self.num_spans,
            span_strategy=self.span_strategy,
            max_seq_length=self.max_seq_length,
        )

        result: Dict[str, torch.Tensor] = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

        if "boundary_positions" in sample:
            result["boundary_positions"] = sample["boundary_positions"]

        return result


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of sample dictionaries from the dataset.
        pad_token_id: Token id used for padding.

    Returns:
        Batched and padded tensors.
    """
    max_len = max(s["input_ids"].size(0) for s in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    boundary_positions_list = []
    has_boundary = "boundary_positions" in batch[0]

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids_list.append(
            torch.cat([sample["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        # Pad attention_mask
        attention_mask_list.append(
            torch.cat([sample["attention_mask"], torch.zeros(pad_len)])
        )
        # Pad labels with -100 (ignore index)
        labels_list.append(
            torch.cat([sample["labels"], torch.full((pad_len,), -100)])
        )

        if has_boundary:
            boundary_positions_list.append(sample["boundary_positions"])

    result: Dict[str, torch.Tensor] = {
        "input_ids": torch.stack(input_ids_list).long(),
        "attention_mask": torch.stack(attention_mask_list).long(),
        "labels": torch.stack(labels_list).long(),
    }

    if has_boundary and boundary_positions_list:
        # Pad boundary_positions to same length (use 0 as pad value)
        max_bp_len = max(bp.size(0) for bp in boundary_positions_list)
        padded_bp = []
        for bp in boundary_positions_list:
            pad_len_bp = max_bp_len - bp.size(0)
            if pad_len_bp > 0:
                padded_bp.append(torch.cat([bp, torch.zeros(pad_len_bp, dtype=bp.dtype)]))
            else:
                padded_bp.append(bp)
        result["boundary_positions"] = torch.stack(padded_bp).long()

    return result
