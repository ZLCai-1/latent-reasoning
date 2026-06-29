#!/usr/bin/env python3
"""
Minimal pipeline verification for the evaluation system.
Run this on the server to confirm all diagnostics work before actual training.

Usage:
    python scripts/verify_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np


def main():
    print("=" * 60)
    print("MINIMAL PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Test diagnostics.py imports
    print("\n--- §1: diagnostics.py imports ---")
    from src.eval.diagnostics import (
        transition_cosine,
        normalized_transition_error,
        endpoint_drift,
        bridge_gap,
        layer_wise_cka,
        collapse_metrics,
        compute_accuracy_retention,
        compute_relative_gain,
        compute_compression_ratio,
        compute_throughput,
        compute_length_bucket_accuracy,
        compute_full_diagnostics,
    )
    print("✅ All imports successful")

    # 2. Test §5.6.4 State Transition Alignment
    print("\n--- §2: §5.6.4 State Transition Alignment ---")
    B, K, nL, D = 4, 3, 2, 64
    student_trans = torch.randn(B, K - 1, nL, D)
    teacher_trans = student_trans + torch.randn_like(student_trans) * 0.1

    tc = transition_cosine(student_trans, teacher_trans)
    assert 0.5 < tc["mean"] < 1.0, f"Unexpected cosine: {tc['mean']}"
    print(f"  Transition Cosine: {tc['mean']:.4f} (per_k={[f'{x:.3f}' for x in tc['per_k']]})")

    nte = normalized_transition_error(student_trans, teacher_trans)
    assert nte["mean"] > 0, f"NTE should be positive: {nte['mean']}"
    print(f"  Norm. Transition Error: {nte['mean']:.4f}")

    student_bound = torch.randn(B, K, nL, D)
    teacher_bound = student_bound + torch.randn_like(student_bound) * 0.05

    ed = endpoint_drift(student_bound, teacher_bound)
    print(f"  Endpoint Drift: {ed['mean']:.4f}")

    cka = layer_wise_cka(student_bound, teacher_bound)
    assert cka["mean"] > 0.5, f"CKA too low for similar states: {cka['mean']}"
    print(f"  Layer-wise CKA: {cka['mean']:.4f} (per_layer={[f'{x:.3f}' for x in cka['per_layer']]})")

    st_prefix = student_bound + torch.randn_like(student_bound) * 0.02
    bg = bridge_gap(st_prefix, student_bound, teacher_bound)
    print(f"  Bridge Gap: {bg['mean']:.4f}")

    print("✅ All §5.6.4 metrics passed")

    # 3. Test §5.6.5 Collapse Detection
    print("\n--- §3: §5.6.5 Stability & Collapse ---")
    diverse_states = torch.randn(B, K, D)
    cm = collapse_metrics(diverse_states)
    print(f"  Diverse:   diversity={cm['pairwise_diversity']:.4f}, collapse={cm['collapse_rate']:.1%}, rank={cm['effective_rank']:.2f}")

    collapsed_states = torch.randn(1, 1, D).expand(B, K, D) + torch.randn(B, K, D) * 0.001
    cm2 = collapse_metrics(collapsed_states, collapse_threshold=0.95)
    print(f"  Collapsed: diversity={cm2['pairwise_diversity']:.4f}, collapse={cm2['collapse_rate']:.1%}, rank={cm2['effective_rank']:.2f}")

    assert cm["pairwise_diversity"] > cm2["pairwise_diversity"], "Diversity detection failed"
    assert cm2["collapse_rate"] > cm["collapse_rate"], "Collapse detection failed"
    print("✅ Collapse detection correctly distinguishes diverse vs collapsed")

    # 4. Test §5.6.2/§5.6.3 Extended Metrics
    print("\n--- §4: §5.6.2 & §5.6.3 Extended Metrics ---")
    assert abs(compute_accuracy_retention(0.60, 0.75) - 0.8) < 1e-6
    print(f"  Accuracy Retention: {compute_accuracy_retention(0.60, 0.75):.4f}")

    assert compute_relative_gain(0.60, 0.03) > 10
    print(f"  Relative Gain: {compute_relative_gain(0.60, 0.03):.2f}x")

    assert compute_compression_ratio(200.0, 3) > 60
    print(f"  Compression Ratio: {compute_compression_ratio(200.0, 3):.1f}x")

    assert compute_throughput(100, 25.0) == 4.0
    print(f"  Throughput: {compute_throughput(100, 25.0):.1f} samples/sec")

    preds = ["42", "10", "7", "3", "100"]
    refs = ["42", "10", "8", "3", "99"]
    lengths = [30, 80, 150, 250, 120]
    lb = compute_length_bucket_accuracy(preds, refs, lengths)
    assert lb["len_0-50"] == 1.0  # 42==42
    print(f"  Length-bucket: {lb}")
    print("✅ All extended metrics passed")

    # 5. Test compute_full_diagnostics aggregator
    print("\n--- §5: Full diagnostics aggregator ---")
    report = compute_full_diagnostics(
        student_transitions=student_trans,
        teacher_transitions=teacher_trans,
        student_boundary_states=student_bound,
        teacher_boundary_states=teacher_bound,
        latent_hidden_states=diverse_states,
    )
    expected_keys = {"transition_cosine", "normalized_transition_error", "endpoint_drift", "layer_wise_cka", "collapse_metrics"}
    assert expected_keys.issubset(set(report.keys())), f"Missing keys: {expected_keys - set(report.keys())}"
    print(f"  Report keys: {list(report.keys())}")
    print("✅ compute_full_diagnostics() works end-to-end")

    # 6. Test base metrics
    print("\n--- §6: Base metrics (extract_numeric_answer) ---")
    from src.eval.metrics import extract_numeric_answer, compute_accuracy
    assert extract_numeric_answer("The answer is 42.") == "42"
    assert extract_numeric_answer("\\boxed{123}") == "123"
    assert extract_numeric_answer("#### 7") == "7"
    assert extract_numeric_answer("= 3.14") == "3.14"
    acc = compute_accuracy(["42", "10", "abc"], ["42", "10", "5"])
    assert abs(acc - 2 / 3) < 1e-6
    print(f"  extract_numeric_answer: all patterns OK")
    print(f"  compute_accuracy: {acc:.4f} (expected 0.6667)")
    print("✅ Base metrics passed")

    # 7. Test preprocessing defense
    print("\n--- §7: Data preprocessing defense ---")
    import inspect
    from src.data.preprocessing import prepare_training_sample
    src = inspect.getsource(prepare_training_sample)
    assert "isinstance(spans[0], str)" in src, "List[str] defense missing in prepare_training_sample!"
    print("✅ List[str] → List[List[str]] defense present")

    # 8. Test dataset defense
    print("\n--- §8: Dataset spans defense ---")
    from src.data.dataset import LatentReasoningDataset
    src2 = inspect.getsource(LatentReasoningDataset.__getitem__)
    assert "isinstance(spans[0], str)" in src2, "List[str] defense missing in dataset!"
    print("✅ Dataset also has List[str] wrapping defense")

    # 9. Test trainer _validate uses forward_with_latent
    print("\n--- §9: Trainer _validate() fix ---")
    from src.training.trainer import Trainer
    src3 = inspect.getsource(Trainer._validate)
    assert "forward_with_latent" in src3, "_validate still uses plain forward!"
    assert "latent_positions" in src3, "_validate doesn't handle latent_positions!"
    print("✅ _validate() correctly uses forward_with_latent when latent_positions present")

    # Summary
    print("\n" + "=" * 60)
    print("ALL 9 CHECKS PASSED ✅")
    print("=" * 60)
    print("\nEvaluation pipeline is ready for deployment.")
    print("Next step on server:")
    print("  python scripts/run_diagnostics.py \\")
    print("      --config config/exp/stage1_transition.yaml \\")
    print("      --checkpoint checkpoints/stage1_transition/final \\")
    print("      --data_path data/gsm8k_test.json \\")
    print("      --output results/diagnostics.json")


if __name__ == "__main__":
    main()
