"""
Evaluation Metrics for Latent Reasoning.

Provides accuracy, exact-match, and numeric-answer extraction utilities
tailored to math-reasoning benchmarks like GSM8K.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a model's generation.

    Handles formats such as:
    - ``#### 42``
    - ``The answer is 42.``
    - ``= 42``
    - Plain trailing number

    Args:
        text: Generated text string.

    Returns:
        Extracted numeric string, or ``None`` if no number is found.
    """
    # Try GSM8K-style "#### <number>"
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return _clean_number(match.group(1))

    # Try "The answer is <number>"
    match = re.search(r"(?:the answer is|answer is|answer:)\s*([\-\d,\.]+)", text, re.IGNORECASE)
    if match:
        return _clean_number(match.group(1))

    # Try "= <number>" at the end
    match = re.search(r"=\s*([\-\d,\.]+)\s*$", text)
    if match:
        return _clean_number(match.group(1))

    # Fallback: last number in text
    numbers = re.findall(r"[\-\d,\.]+", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None


def _clean_number(s: str) -> str:
    """Remove commas and trailing dots from a numeric string."""
    s = s.replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    return s


def compute_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute numeric accuracy between predictions and references.

    Both lists should contain the *final numeric answer* strings.
    Comparison is done after normalising with :func:`_clean_number`.

    Args:
        predictions: Model-predicted answer strings.
        references: Ground-truth answer strings.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = extract_numeric_answer(pred) if pred else None
        ref_num = _clean_number(ref) if ref else None
        if pred_num is not None and ref_num is not None and pred_num == ref_num:
            correct += 1

    return correct / len(predictions)


def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute exact string match after basic normalisation.

    Args:
        predictions: Predicted strings.
        references: Reference strings.

    Returns:
        Exact-match rate in ``[0, 1]``.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        if _normalise_text(pred) == _normalise_text(ref):
            correct += 1

    return correct / len(predictions)


def _normalise_text(text: str) -> str:
    """Lowercase, strip whitespace, and remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
