"""
Data Preprocessing for Latent Reasoning.

Handles splitting CoT steps into spans, inserting boundary markers,
and constructing complete training samples with proper labelling.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def split_into_spans(
    steps: List[str],
    num_spans: int = 3,
    strategy: str = "fixed",
) -> List[List[str]]:
    """Partition a list of CoT steps into *num_spans* spans.

    Args:
        steps: Individual reasoning steps.
        num_spans: Target number of spans (``K``).
        strategy:
            - ``"fixed"``: Evenly split steps into K groups.
            - ``"random"``: Randomly merge adjacent steps (data
              augmentation).
            - ``"semantic"``: Placeholder for future semantic-boundary
              splitting.

    Returns:
        Nested list where each sub-list contains the steps belonging
        to that span.
    """
    if not steps:
        return [[]]

    if num_spans <= 0 or num_spans >= len(steps):
        # If K ≥ num_steps, each step is its own span
        return [[s] for s in steps] if steps else [[]]

    if strategy == "fixed":
        return _split_fixed(steps, num_spans)
    elif strategy == "random":
        return _split_random(steps, num_spans)
    elif strategy == "semantic":
        # Placeholder – fall back to fixed for now
        logger.warning("Semantic splitting not yet implemented; using fixed.")
        return _split_fixed(steps, num_spans)
    else:
        raise ValueError(f"Unknown span strategy: {strategy!r}")


def _split_fixed(steps: List[str], num_spans: int) -> List[List[str]]:
    """Evenly distribute *steps* into *num_spans* groups."""
    n = len(steps)
    base_size = n // num_spans
    remainder = n % num_spans

    spans: List[List[str]] = []
    idx = 0
    for i in range(num_spans):
        size = base_size + (1 if i < remainder else 0)
        group = steps[idx : idx + size]
        spans.append(group)
        idx += size
    return spans


def _split_random(steps: List[str], num_spans: int) -> List[List[str]]:
    """Randomly choose split points among *steps* to create *num_spans*."""
    n = len(steps)
    if num_spans >= n:
        return [[s] for s in steps]

    # Choose (num_spans - 1) unique split points from [1, n-1]
    split_points = sorted(random.sample(range(1, n), num_spans - 1))
    split_points = [0] + split_points + [n]

    spans: List[List[str]] = []
    for i in range(len(split_points) - 1):
        group = steps[split_points[i] : split_points[i + 1]]
        spans.append(group)
    return spans


def insert_boundary_markers(
    text: str,
    span_boundaries: List[int],
    tokenizer: Any,
) -> str:
    """Insert ``<SPAN_START>`` / ``<SPAN_END>`` tokens at span boundaries.

    This operates on the *text* level before tokenization.

    Args:
        text: Full text string.
        span_boundaries: Character offsets where boundaries occur.
        tokenizer: Tokenizer instance (used to retrieve special token
                   strings, though not strictly required here).

    Returns:
        Text with boundary markers inserted.
    """
    if not span_boundaries:
        return text

    # Sort in reverse so that insertions don't shift later offsets
    sorted_boundaries = sorted(set(span_boundaries), reverse=True)
    result = text
    for pos in sorted_boundaries:
        pos = min(pos, len(result))
        result = result[:pos] + " <SPAN_END> <SPAN_START> " + result[pos:]
    return result


def prepare_training_sample(
    question: str,
    spans: List[List[str]],
    answer: str,
    tokenizer: Any,
    num_spans: int = 3,
    span_strategy: str = "fixed",
    max_seq_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Construct a single training sample with boundary markers.

    The format is::

        <question> <SPAN_START> span_1 <SPAN_END> <SPAN_START> span_2
        <SPAN_END> ... <SPAN_START> span_K <SPAN_END> <answer>

    For ``span_strategy="none"`` (Stage 0 CoT training), no boundaries
    are inserted and the full CoT is used.

    Args:
        question: Problem text.
        spans: Pre-computed spans as ``List[List[str]]`` (each sub-list is
               a group of steps, joined internally as span text).
        answer: Final answer string.
        tokenizer: Tokenizer with boundary special tokens.
        num_spans: Number of target spans.
        span_strategy: Splitting strategy (``"fixed"`` / ``"random"`` /
                       ``"none"``).
        max_seq_length: Maximum token length.

    Returns:
        Dictionary with ``input_ids``, ``attention_mask``, ``labels``,
        and optionally ``boundary_positions`` (tensor of boundary token
        positions).
    """
    if span_strategy == "none" or num_spans == 0:
        # Stage 0: plain CoT without boundaries
        cot_text = " ".join(" ".join(group) for group in spans)
        full_text = f"Question: {question}\nAnswer: {cot_text} {answer}"
        return _tokenize_and_label(full_text, tokenizer, max_seq_length)

    # Build text with boundary markers
    parts = [f"Question: {question}\nAnswer:"]
    for span_group in spans:
        span_text = " ".join(span_group)
        parts.append(f" <SPAN_START> {span_text} <SPAN_END>")
    parts.append(f" {answer}")
    full_text = "".join(parts)

    return _tokenize_and_label(
        full_text,
        tokenizer,
        max_seq_length,
        find_boundaries=True,
    )


def _tokenize_and_label(
    text: str,
    tokenizer: Any,
    max_seq_length: int,
    find_boundaries: bool = False,
) -> Dict[str, torch.Tensor]:
    """Tokenize text and create causal-LM labels.

    Labels are set to -100 for the question portion so that only the
    answer/CoT part contributes to the generation loss.

    Args:
        text: Full input text.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Maximum sequence length.
        find_boundaries: Whether to locate ``<SPAN_START>`` /
                         ``<SPAN_END>`` positions.

    Returns:
        Dictionary with tensors.
    """
    encoding = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)  # [L]
    attention_mask = encoding["attention_mask"].squeeze(0)  # [L]

    # Labels: copy of input_ids (causal LM objective)
    labels = input_ids.clone()

    # Mask question portion (everything before "Answer:") with -100
    answer_token_ids = tokenizer.encode("Answer:", add_special_tokens=False)
    answer_start = _find_subseq(input_ids.tolist(), answer_token_ids)
    if answer_start is not None:
        labels[:answer_start] = -100

    result: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if find_boundaries:
        span_start_id = tokenizer.convert_tokens_to_ids("<SPAN_START>")
        span_end_id = tokenizer.convert_tokens_to_ids("<SPAN_END>")

        boundary_positions = []
        ids_list = input_ids.tolist()
        for i, tid in enumerate(ids_list):
            if tid in (span_start_id, span_end_id):
                boundary_positions.append(i)

        if boundary_positions:
            result["boundary_positions"] = torch.tensor(
                boundary_positions, dtype=torch.long
            )

    return result


def prepare_student_sample(
    question: str,
    spans: List[List[str]],
    answer: str,
    tokenizer: Any,
    num_latent_tokens: int = 3,
    max_seq_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Construct a Student training sample: replace CoT spans with latent tokens.

    The format is::

        Question: <question>\nAnswer: <LATENT> <LATENT> ... <LATENT> <answer>

    All CoT span content is removed and replaced by *num_latent_tokens*
    ``<LATENT>`` placeholder tokens.  The model's ``forward_with_latent``
    method will inject learned embeddings at these positions.

    Args:
        question: Problem text.
        spans: Pre-computed spans (used only to determine original CoT
               existed; content is discarded).
        answer: Final answer string.
        tokenizer: Tokenizer with ``<LATENT>`` special token added.
        num_latent_tokens: Number of latent tokens to insert (``K``).
        max_seq_length: Maximum token length.

    Returns:
        Dictionary with:
          - ``input_ids``: Token ids ``[L]``.
          - ``attention_mask``: Mask ``[L]``.
          - ``labels``: Labels ``[L]`` (question masked with -100,
            latent positions masked with -100, only answer contributes).
          - ``latent_positions``: ``[K]`` int tensor with the positions
            of the latent tokens in the sequence.
          - ``answer_start``: Scalar int tensor marking where the answer
            begins (for generation loss).
    """
    # Build student text: Question + latent placeholders + Answer
    latent_str = " ".join(["<LATENT>"] * num_latent_tokens)
    full_text = f"Question: {question}\nAnswer: {latent_str} {answer}"

    encoding = tokenizer(
        full_text,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)  # [L]
    attention_mask = encoding["attention_mask"].squeeze(0)  # [L]

    # Labels: copy of input_ids (causal LM objective)
    labels = input_ids.clone()

    # Find latent token positions
    latent_token_id = tokenizer.convert_tokens_to_ids("<LATENT>")
    ids_list = input_ids.tolist()
    latent_positions = [i for i, tid in enumerate(ids_list) if tid == latent_token_id]

    # Mask question portion (everything before "Answer:") with -100
    answer_token_ids = tokenizer.encode("Answer:", add_special_tokens=False)
    answer_start = _find_subseq(ids_list, answer_token_ids)
    if answer_start is not None:
        labels[:answer_start] = -100

    # Mask latent positions in labels (latent tokens should not contribute
    # to generation loss — they have no ground-truth next token)
    for pos in latent_positions:
        labels[pos] = -100

    # Determine where the actual answer text starts (after latent tokens)
    if latent_positions:
        answer_text_start = latent_positions[-1] + 1
    elif answer_start is not None:
        answer_text_start = answer_start
    else:
        answer_text_start = 0

    result: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "latent_positions": torch.tensor(latent_positions, dtype=torch.long),
        "answer_start": torch.tensor(answer_text_start, dtype=torch.long),
    }

    return result


def _find_subseq(seq: List[int], subseq: List[int]) -> Optional[int]:
    """Find the start index of *subseq* within *seq*."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return None


