from .evaluator import Evaluator
from .metrics import compute_accuracy, compute_exact_match, extract_numeric_answer
from .diagnostics import (
    transition_cosine,
    normalized_transition_error,
    endpoint_drift,
    bridge_gap,
    layer_wise_cka,
    collapse_metrics,
    causal_intervention,
    compute_accuracy_retention,
    compute_relative_gain,
    compute_compression_ratio,
    compute_throughput,
    compute_length_bucket_accuracy,
    compute_full_diagnostics,
)

__all__ = [
    "Evaluator",
    "compute_accuracy",
    "compute_exact_match",
    "extract_numeric_answer",
    # §5.6.4 State Transition Alignment
    "transition_cosine",
    "normalized_transition_error",
    "endpoint_drift",
    "bridge_gap",
    "layer_wise_cka",
    # §5.6.5 Stability & Interpretability
    "collapse_metrics",
    "causal_intervention",
    # §5.6.2 & §5.6.3 Extended Metrics
    "compute_accuracy_retention",
    "compute_relative_gain",
    "compute_compression_ratio",
    "compute_throughput",
    "compute_length_bucket_accuracy",
    # Aggregated
    "compute_full_diagnostics",
]
