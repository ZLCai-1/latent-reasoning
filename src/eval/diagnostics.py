"""
Diagnostic Metrics for Latent Reasoning (Research Proposal §5.6.4 & §5.6.5).

Implements state-transition alignment metrics, stability/collapse detection,
and interpretability diagnostics for evaluating latent reasoning quality
beyond simple task accuracy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# §5.6.4 State Transition Alignment Metrics
# =============================================================================


def transition_cosine(
    student_transitions: torch.Tensor,
    teacher_transitions: torch.Tensor,
) -> Dict[str, float]:
    """Compute cosine similarity between student and teacher transitions.

    Measures whether student ΔH direction matches teacher ΔH direction.

    Args:
        student_transitions: ``[B, K-1, num_layers, D]`` or ``[B, K-1, D]``.
        teacher_transitions: Same shape as student.

    Returns:
        Dictionary with ``mean``, ``per_k`` (list), and ``per_layer`` (list if 4D).
    """
    # Align K dimension
    K_min = min(student_transitions.size(1), teacher_transitions.size(1))
    if K_min == 0:
        return {"mean": 0.0, "per_k": [], "per_layer": []}
    s = student_transitions[:, :K_min].float()
    t = teacher_transitions[:, :K_min].float()

    # Flatten to compute cosine per (batch, k) pair
    if s.dim() == 4:
        # [B, K-1, nL, D] → compute per layer
        B, K, nL, D = s.shape
        per_layer_cos = []
        for l in range(nL):
            s_l = s[:, :, l, :]  # [B, K, D]
            t_l = t[:, :, l, :]
            cos = F.cosine_similarity(s_l, t_l, dim=-1)  # [B, K]
            per_layer_cos.append(cos.mean().item())

        # Overall
        s_flat = s.reshape(B * K, nL * D)
        t_flat = t.reshape(B * K, nL * D)
        overall_cos = F.cosine_similarity(s_flat, t_flat, dim=-1)

        # Per-k
        s_k = s.reshape(B, K, nL * D)
        t_k = t.reshape(B, K, nL * D)
        per_k_cos = F.cosine_similarity(s_k, t_k, dim=-1).mean(dim=0)  # [K]

        return {
            "mean": overall_cos.mean().item(),
            "per_k": per_k_cos.tolist(),
            "per_layer": per_layer_cos,
        }
    else:
        # [B, K-1, D]
        cos = F.cosine_similarity(s, t, dim=-1)  # [B, K]
        per_k = cos.mean(dim=0).tolist()
        return {
            "mean": cos.mean().item(),
            "per_k": per_k,
        }


def normalized_transition_error(
    student_transitions: torch.Tensor,
    teacher_transitions: torch.Tensor,
) -> Dict[str, float]:
    """Compute normalized transition error: ‖ΔS - ΔT‖ / ‖ΔT‖.

    Measures combined magnitude and direction error of state updates.

    Args:
        student_transitions: ``[B, K-1, ...]`` student ΔH.
        teacher_transitions: ``[B, K-1, ...]`` teacher ΔH.

    Returns:
        Dictionary with ``mean``, ``per_k`` (list).
    """
    K_min = min(student_transitions.size(1), teacher_transitions.size(1))
    if K_min == 0:
        return {"mean": 0.0, "per_k": []}
    s = student_transitions[:, :K_min].float()
    t = teacher_transitions[:, :K_min].float()

    # Flatten last dims for norm computation
    orig_shape = s.shape
    B, K = orig_shape[0], orig_shape[1]
    s_flat = s.reshape(B, K, -1)
    t_flat = t.reshape(B, K, -1)

    diff_norm = torch.norm(s_flat - t_flat, dim=-1)  # [B, K]
    t_norm = torch.norm(t_flat, dim=-1).clamp(min=1e-8)  # [B, K]

    nte = diff_norm / t_norm  # [B, K]
    per_k = nte.mean(dim=0).tolist()

    return {
        "mean": nte.mean().item(),
        "per_k": per_k,
    }


def endpoint_drift(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
) -> Dict[str, float]:
    """Compute endpoint drift: ‖s_k^S - s_k^T‖ / ‖s_k^T‖.

    Checks whether aligning only transitions causes overall state drift.

    Args:
        student_states: ``[B, K, ...]`` student boundary states.
        teacher_states: ``[B, K, ...]`` teacher boundary states.

    Returns:
        Dictionary with ``mean``, ``per_k`` (list).
    """
    K_min = min(student_states.size(1), teacher_states.size(1))
    s = student_states[:, :K_min].float()
    t = teacher_states[:, :K_min].float()

    B, K = s.shape[0], s.shape[1]
    s_flat = s.reshape(B, K, -1)
    t_flat = t.reshape(B, K, -1)

    diff_norm = torch.norm(s_flat - t_flat, dim=-1)  # [B, K]
    t_norm = torch.norm(t_flat, dim=-1).clamp(min=1e-8)  # [B, K]

    drift = diff_norm / t_norm  # [B, K]
    per_k = drift.mean(dim=0).tolist()

    return {
        "mean": drift.mean().item(),
        "per_k": per_k,
    }


def bridge_gap(
    student_states_teacher_prefix: torch.Tensor,
    student_states_self_prefix: torch.Tensor,
    teacher_states: torch.Tensor,
) -> Dict[str, float]:
    """Compute bridge gap: ‖ŝ^{S←T} - ŝ^{S←S}‖ / ‖s^T‖.

    Verifies consistency between teacher-predecessor and student-rollout.

    Args:
        student_states_teacher_prefix: ``[B, K, ...]`` ŝ^{S←T}.
        student_states_self_prefix: ``[B, K, ...]`` ŝ^{S←S}.
        teacher_states: ``[B, K, ...]`` s^T.

    Returns:
        Dictionary with ``mean``, ``per_k`` (list).
    """
    K_min = min(
        student_states_teacher_prefix.size(1),
        student_states_self_prefix.size(1),
        teacher_states.size(1),
    )
    st = student_states_teacher_prefix[:, :K_min].float()
    ss = student_states_self_prefix[:, :K_min].float()
    t = teacher_states[:, :K_min].float()

    B, K = st.shape[0], st.shape[1]
    st_flat = st.reshape(B, K, -1)
    ss_flat = ss.reshape(B, K, -1)
    t_flat = t.reshape(B, K, -1)

    gap_norm = torch.norm(st_flat - ss_flat, dim=-1)  # [B, K]
    t_norm = torch.norm(t_flat, dim=-1).clamp(min=1e-8)

    gap = gap_norm / t_norm
    per_k = gap.mean(dim=0).tolist()

    return {
        "mean": gap.mean().item(),
        "per_k": per_k,
    }


def layer_wise_cka(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
) -> Dict[str, float]:
    """Compute linear CKA between student and teacher boundary states per layer.

    CKA (Centered Kernel Alignment) measures representational similarity
    regardless of rotation/scaling.

    Args:
        student_states: ``[B, K, num_layers, D]``.
        teacher_states: ``[B, K, num_layers, D]``.

    Returns:
        Dictionary with ``per_layer`` (list of CKA values).
    """
    K_min = min(student_states.size(1), teacher_states.size(1))
    s = student_states[:, :K_min].float()
    t = teacher_states[:, :K_min].float()

    if s.dim() != 4:
        # Cannot compute per-layer if not 4D
        return {"per_layer": [], "mean": 0.0}

    B, K, nL, D = s.shape
    per_layer_cka = []

    for l in range(nL):
        # Reshape to [B*K, D] for CKA
        X = s[:, :, l, :].reshape(-1, D)
        Y = t[:, :, l, :].reshape(-1, D)
        cka_val = _linear_cka(X, Y)
        per_layer_cka.append(cka_val)

    return {
        "per_layer": per_layer_cka,
        "mean": float(np.mean(per_layer_cka)) if per_layer_cka else 0.0,
    }


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: ``[N, D1]`` representations.
        Y: ``[N, D2]`` representations.

    Returns:
        CKA similarity value in [0, 1].
    """
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices (linear kernel)
    # CKA = ‖Y^T X‖_F^2 / (‖X^T X‖_F * ‖Y^T Y‖_F)
    XtX = (X.T @ X)  # [D1, D1]
    YtY = (Y.T @ Y)  # [D2, D2]
    YtX = (Y.T @ X)  # [D2, D1]

    hsic_xy = (YtX * YtX).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-10)
    return (hsic_xy / denom).item()


# =============================================================================
# §5.6.5 Stability, Collapse & Interpretability Metrics
# =============================================================================


def collapse_metrics(
    latent_hidden_states: torch.Tensor,
    collapse_threshold: float = 0.95,
) -> Dict[str, float]:
    """Compute latent token collapse and diversity metrics.

    Detects whether latent tokens have collapsed into homogeneous
    representations by measuring pairwise cosine similarity and
    effective rank.

    Args:
        latent_hidden_states: ``[B, K, D]`` hidden states at latent
            token positions across the batch.
        collapse_threshold: Cosine similarity threshold above which
            a pair is considered collapsed.

    Returns:
        Dictionary with:
          - ``pairwise_diversity``: mean (1 - cosine_sim) across pairs.
          - ``collapse_rate``: fraction of pairs above threshold.
          - ``effective_rank``: effective rank of representation covariance.
          - ``mean_cosine``: mean pairwise cosine similarity.
    """
    if latent_hidden_states.dim() == 4:
        # [B, K, nL, D] → flatten layers
        B, K, nL, D = latent_hidden_states.shape
        latent_hidden_states = latent_hidden_states.reshape(B, K, nL * D)
    else:
        B, K, D = latent_hidden_states.shape

    states = latent_hidden_states.float()  # [B, K, D']
    D_flat = states.size(-1)

    # Compute pairwise cosine between latent tokens within each sample
    # Normalize
    states_norm = F.normalize(states, dim=-1)  # [B, K, D']

    all_cosines = []
    for b in range(B):
        # [K, D'] → pairwise cosine [K, K]
        cos_matrix = states_norm[b] @ states_norm[b].T  # [K, K]
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(K, K, device=states.device), diagonal=1).bool()
        pairwise = cos_matrix[mask]
        all_cosines.append(pairwise)

    if all_cosines:
        all_cos = torch.cat(all_cosines)
        mean_cosine = all_cos.mean().item()
        pairwise_diversity = 1.0 - mean_cosine
        collapse_rate = (all_cos > collapse_threshold).float().mean().item()
    else:
        mean_cosine = 0.0
        pairwise_diversity = 1.0
        collapse_rate = 0.0

    # Effective rank: from singular values of [B*K, D] matrix
    all_states = states.reshape(-1, D_flat)  # [B*K, D']
    # Center
    all_states = all_states - all_states.mean(dim=0, keepdim=True)

    # SVD for effective rank (use min(N, D) singular values)
    try:
        # Use truncated SVD for efficiency
        n_components = min(all_states.size(0), all_states.size(1), 64)
        U, S, V = torch.svd_lowrank(all_states, q=n_components)
        # Effective rank = exp(entropy of normalized singular values)
        s_norm = S / S.sum().clamp(min=1e-10)
        s_norm = s_norm[s_norm > 1e-10]  # filter zeros
        entropy = -(s_norm * torch.log(s_norm)).sum()
        eff_rank = torch.exp(entropy).item()
    except Exception:
        eff_rank = float(K)

    return {
        "pairwise_diversity": pairwise_diversity,
        "collapse_rate": collapse_rate,
        "effective_rank": eff_rank,
        "mean_cosine": mean_cosine,
    }


def causal_intervention(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_positions: torch.Tensor,
    labels: torch.Tensor,
    num_latent_tokens: int,
) -> Dict[str, List[float]]:
    """Causal intervention: measure accuracy drop when each z_k is removed/shuffled.

    For each latent token position k:
    1. Zero out its embedding and measure accuracy change.
    2. Shuffle its position with another token.

    This tests whether each latent token has a distinct functional role.

    Args:
        model: The LatentReasoningModel.
        input_ids: ``[B, L]`` input token ids.
        attention_mask: ``[B, L]`` attention mask.
        latent_positions: ``[B, K]`` latent token positions.
        labels: ``[B, L]`` labels for loss computation.
        num_latent_tokens: Number of latent tokens K.

    Returns:
        Dictionary with ``zero_out_loss_increase`` per k.
    """
    from ..models.loss_functions import generation_loss

    device = input_ids.device
    K = latent_positions.size(1)

    # Baseline: normal forward with all latent embeddings
    with torch.no_grad():
        baseline_outputs = model.forward_with_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_positions=latent_positions,
        )
        baseline_loss = generation_loss(baseline_outputs["logits"], labels).item()

    # Zero-out each latent token k and measure loss increase
    zero_out_increases = []
    for k in range(min(K, num_latent_tokens)):
        with torch.no_grad():
            # Get base embeddings
            inputs_embeds = model.model.get_input_embeddings()(input_ids).clone()

            # Inject all latent embeddings EXCEPT k
            for kk in range(min(K, num_latent_tokens)):
                if kk == k:
                    continue  # Skip this one (leave as original token embedding)
                latent_emb = model.latent_embeddings(
                    torch.tensor(kk, device=device)
                )
                for b in range(input_ids.size(0)):
                    pos = latent_positions[b, kk].item()
                    seq_len = inputs_embeds.size(1)
                    if 0 < pos < seq_len:
                        inputs_embeds[b, pos, :] = latent_emb

            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            loss_k = generation_loss(outputs["logits"], labels).item()
            zero_out_increases.append(loss_k - baseline_loss)

    return {
        "baseline_loss": baseline_loss,
        "zero_out_loss_increase": zero_out_increases,
    }


# =============================================================================
# §5.6.2 & §5.6.3 Extended Task & Efficiency Metrics
# =============================================================================


def compute_accuracy_retention(
    latent_accuracy: float,
    cot_accuracy: float,
) -> float:
    """Accuracy Retention = Acc_latent / Acc_CoT.

    Args:
        latent_accuracy: Accuracy of the latent student model.
        cot_accuracy: Accuracy of the explicit CoT teacher.

    Returns:
        Retention ratio (1.0 = perfect retention).
    """
    if cot_accuracy <= 0:
        return 0.0
    return latent_accuracy / cot_accuracy


def compute_relative_gain(
    latent_accuracy: float,
    direct_accuracy: float,
) -> float:
    """Relative Gain over Direct = (Acc_latent - Acc_direct) / Acc_direct.

    Proves latent token is not degenerating into direct answer.

    Args:
        latent_accuracy: Accuracy of the latent student.
        direct_accuracy: Accuracy of direct answer baseline.

    Returns:
        Relative gain ratio.
    """
    if direct_accuracy <= 0:
        return 0.0
    return (latent_accuracy - direct_accuracy) / direct_accuracy


def compute_compression_ratio(
    avg_cot_tokens: float,
    num_latent_tokens: int,
) -> float:
    """Reasoning Compression Ratio = teacher_CoT_tokens / K.

    Args:
        avg_cot_tokens: Average number of CoT output tokens.
        num_latent_tokens: Number of latent tokens K.

    Returns:
        Compression ratio (how many CoT tokens per latent token).
    """
    if num_latent_tokens <= 0:
        return 0.0
    return avg_cot_tokens / num_latent_tokens


def compute_throughput(
    num_samples: int,
    total_time_seconds: float,
) -> float:
    """Throughput in samples/sec.

    Args:
        num_samples: Total number of samples processed.
        total_time_seconds: Total wall-clock time.

    Returns:
        Throughput (samples per second).
    """
    if total_time_seconds <= 0:
        return 0.0
    return num_samples / total_time_seconds


def compute_length_bucket_accuracy(
    predictions: List[str],
    references: List[str],
    cot_lengths: List[int],
    buckets: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Compute accuracy per CoT length bucket.

    Args:
        predictions: Model predictions.
        references: Ground truth answers.
        cot_lengths: Number of CoT tokens for each sample.
        buckets: List of (min_len, max_len) tuples. Defaults to
                 [(0,50), (50,100), (100,200), (200,inf)].

    Returns:
        Dictionary mapping bucket name to accuracy.
    """
    from .metrics import extract_numeric_answer

    if buckets is None:
        buckets = [(0, 50), (50, 100), (100, 200), (200, 999999)]

    bucket_results: Dict[str, Dict[str, int]] = {}
    for bmin, bmax in buckets:
        name = f"len_{bmin}-{bmax}"
        bucket_results[name] = {"correct": 0, "total": 0}

    for pred, ref, length in zip(predictions, references, cot_lengths):
        pred_num = extract_numeric_answer(pred) if pred else None
        ref_num = extract_numeric_answer(ref) if ref else None
        is_correct = (
            pred_num is not None and ref_num is not None and pred_num == ref_num
        )

        for bmin, bmax in buckets:
            if bmin <= length < bmax:
                name = f"len_{bmin}-{bmax}"
                bucket_results[name]["total"] += 1
                if is_correct:
                    bucket_results[name]["correct"] += 1
                break

    result = {}
    for name, counts in bucket_results.items():
        if counts["total"] > 0:
            result[name] = counts["correct"] / counts["total"]
        else:
            result[name] = 0.0
        result[f"{name}_count"] = float(counts["total"])

    return result


# =============================================================================
# Aggregated Diagnostic Report
# =============================================================================


def compute_full_diagnostics(
    student_transitions: Optional[torch.Tensor] = None,
    teacher_transitions: Optional[torch.Tensor] = None,
    student_boundary_states: Optional[torch.Tensor] = None,
    teacher_boundary_states: Optional[torch.Tensor] = None,
    student_states_teacher_prefix: Optional[torch.Tensor] = None,
    latent_hidden_states: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute all available diagnostic metrics in one call.

    Pass whatever tensors are available; metrics requiring missing
    tensors will be skipped.

    Returns:
        Nested dictionary with all computed metrics.
    """
    report: Dict[str, Any] = {}

    # §5.6.4 Transition alignment
    if student_transitions is not None and teacher_transitions is not None:
        report["transition_cosine"] = transition_cosine(
            student_transitions, teacher_transitions
        )
        report["normalized_transition_error"] = normalized_transition_error(
            student_transitions, teacher_transitions
        )

    if student_boundary_states is not None and teacher_boundary_states is not None:
        report["endpoint_drift"] = endpoint_drift(
            student_boundary_states, teacher_boundary_states
        )
        report["layer_wise_cka"] = layer_wise_cka(
            student_boundary_states, teacher_boundary_states
        )

    if (
        student_states_teacher_prefix is not None
        and student_boundary_states is not None
        and teacher_boundary_states is not None
    ):
        report["bridge_gap"] = bridge_gap(
            student_states_teacher_prefix,
            student_boundary_states,
            teacher_boundary_states,
        )

    # §5.6.5 Collapse & diversity
    if latent_hidden_states is not None:
        report["collapse_metrics"] = collapse_metrics(latent_hidden_states)

    return report
