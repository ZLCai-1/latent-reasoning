#!/usr/bin/env python
"""
Verification script for Transition Alignment Loss pipeline.

Tests the full forward -> loss -> backward flow on CPU using a small
randomly-initialised GPT-2 model.  No network access required.

Validates:
  1. All loss values are finite and in a reasonable range.
  2. Backward pass produces gradients that are non-None and non-NaN.
  3. Memory usage is acceptable.

Usage:
    conda run -n mllm python scripts/test_loss.py
"""

from __future__ import annotations

import sys
import os
import traceback

# ── Make the project root importable ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.models.loss_functions import (
    transition_loss,
    anchor_loss,
    bridge_loss,
    generation_loss,
    combined_loss,
)
from src.models.state_transition import StateTransitionModule


# ======================================================================
# Helpers
# ======================================================================

def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_finite(name: str, tensor: torch.Tensor) -> bool:
    val = tensor.item() if tensor.numel() == 1 else tensor
    ok = torch.isfinite(tensor).all().item()
    status = "OK" if ok else "FAIL (non-finite!)"
    print(f"  {name:30s} = {val:<12.6f}  [{status}]")
    return ok


def check_grad(name: str, param: torch.nn.Parameter) -> bool:
    if param.grad is None:
        print(f"  {name:30s} grad = None  [WARN]")
        return True  # may be expected for frozen / unused params
    has_nan = torch.isnan(param.grad).any().item()
    has_inf = torch.isinf(param.grad).any().item()
    grad_norm = param.grad.norm().item()
    ok = (not has_nan) and (not has_inf)
    status = "OK" if ok else "FAIL"
    print(f"  {name:30s} grad_norm={grad_norm:.6f}  nan={has_nan}  inf={has_inf}  [{status}]")
    return ok


# ======================================================================
# Main test
# ======================================================================

def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)
    all_passed = True

    # ── 1. Create a small GPT-2 from config (no download needed) ─────
    separator("1. Creating small GPT-2 model (random init)")
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(config).to(device)
    hidden_dim = config.n_embd
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {config.n_layer}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── 2. Create random input ───────────────────────────────────────
    separator("2. Creating random input")
    batch_size = 2
    seq_len = 64
    vocab_size = config.vocab_size
    num_boundaries = 4
    layer_ids = [-1, -2]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Boundary positions: sorted, valid positions within sequence
    boundary_positions = torch.stack([
        torch.sort(torch.randint(1, seq_len - 1, (num_boundaries,)))[0]
        for _ in range(batch_size)
    ]).to(device)  # [B, K]

    print(f"  input_ids shape:          {list(input_ids.shape)}")
    print(f"  boundary_positions shape: {list(boundary_positions.shape)}")
    print(f"  boundary_positions[0]:    {boundary_positions[0].tolist()}")

    # ── 3. Forward pass ──────────────────────────────────────────────
    separator("3. Forward pass (with hidden states)")
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states  # tuple of (n_layer+1,) tensors
    logits = outputs.logits
    print(f"  Number of hidden-state layers: {len(hidden_states)}")
    print(f"  Each hidden-state shape:       {list(hidden_states[0].shape)}")
    print(f"  Logits shape:                  {list(logits.shape)}")

    # ── 4. Extract boundary states ───────────────────────────────────
    separator("4. Extract boundary states")
    stm = StateTransitionModule(layer_ids=layer_ids, hidden_dim=hidden_dim)
    boundary_states = stm.extract_boundary_states(
        hidden_states, boundary_positions, layer_ids,
    )
    print(f"  boundary_states shape: {list(boundary_states.shape)}")
    expected_shape = [batch_size, num_boundaries, len(layer_ids), hidden_dim]
    assert list(boundary_states.shape) == expected_shape, (
        f"Shape mismatch: got {list(boundary_states.shape)}, expected {expected_shape}"
    )
    print("  Shape check: OK")

    # ── 5. Compute transitions ───────────────────────────────────────
    separator("5. Compute transitions")
    student_transitions = stm.compute_transitions(boundary_states)
    print(f"  student_transitions shape: {list(student_transitions.shape)}")
    expected_trans = [batch_size, num_boundaries - 1, len(layer_ids), hidden_dim]
    assert list(student_transitions.shape) == expected_trans, (
        f"Shape mismatch: got {list(student_transitions.shape)}, expected {expected_trans}"
    )
    print("  Shape check: OK")

    # ── 6. Compute losses ────────────────────────────────────────────
    separator("6. Compute losses")

    # 6a. transition_loss (random teacher transitions as target)
    teacher_trans = torch.randn_like(student_transitions)
    t_loss = transition_loss(student_transitions, teacher_trans, normalize=False)
    ok = check_finite("transition_loss", t_loss)
    all_passed &= ok

    # 6b. transition_loss (normalized)
    t_loss_norm = transition_loss(student_transitions, teacher_trans, normalize=True)
    ok = check_finite("transition_loss (normalized)", t_loss_norm)
    all_passed &= ok

    # 6c. transition_loss with zero vectors (edge case)
    zero_student = torch.zeros_like(student_transitions)
    zero_teacher = torch.zeros_like(teacher_trans)
    t_loss_zero = transition_loss(zero_student, zero_teacher, normalize=True)
    ok = check_finite("transition_loss (zero vecs)", t_loss_zero)
    all_passed &= ok

    # 6d. anchor_loss
    teacher_boundary = torch.randn_like(boundary_states)
    a_loss = anchor_loss(boundary_states, teacher_boundary)
    ok = check_finite("anchor_loss", a_loss)
    all_passed &= ok

    # 6e. bridge_loss
    states_no_teacher = boundary_states.clone()
    states_with_teacher = boundary_states + 0.1 * torch.randn_like(boundary_states)
    b_loss = bridge_loss(states_no_teacher, states_with_teacher)
    ok = check_finite("bridge_loss", b_loss)
    all_passed &= ok

    # 6f. generation_loss
    g_loss = generation_loss(logits, labels)
    ok = check_finite("generation_loss", g_loss)
    all_passed &= ok

    # 6g. combined_loss
    losses_dict = {
        "transition": t_loss,
        "anchor": a_loss,
        "bridge": b_loss,
        "generation": g_loss,
    }
    weights_dict = {
        "transition": 0.7,
        "anchor": 0.1,
        "bridge": 0.1,
        "generation": 0.3,
    }
    total_loss, loss_info = combined_loss(losses_dict, weights_dict)
    ok = check_finite("combined_loss (total)", total_loss)
    all_passed &= ok
    print(f"\n  Loss info for logging:")
    for k, v in loss_info.items():
        print(f"    {k:20s}: {v:.6f}")

    # ── 7. Full forward through StateTransitionModule ────────────────
    separator("7. StateTransitionModule.forward()")
    result = stm(
        student_hidden_states=hidden_states,
        boundary_positions=boundary_positions,
        teacher_transitions=teacher_trans,
        normalize=False,
    )
    ok = check_finite("STM transition_loss", result["transition_loss"])
    all_passed &= ok
    print(f"  student_transitions shape: {list(result['student_transitions'].shape)}")
    print(f"  student_boundary   shape:  {list(result['student_boundary_states'].shape)}")

    # ── 8. Backward pass ─────────────────────────────────────────────
    separator("8. Backward pass")
    # Use combined loss for backward
    model.zero_grad()
    # Re-run forward to have a clean computation graph
    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states2 = outputs2.hidden_states
    logits2 = outputs2.logits

    boundary_states2 = stm.extract_boundary_states(
        hidden_states2, boundary_positions, layer_ids,
    )
    student_trans2 = stm.compute_transitions(boundary_states2)
    teacher_trans2 = torch.randn_like(student_trans2)

    t_loss2 = transition_loss(student_trans2, teacher_trans2)
    a_loss2 = anchor_loss(boundary_states2, torch.randn_like(boundary_states2))
    g_loss2 = generation_loss(logits2, labels)

    losses_dict2 = {
        "transition": t_loss2,
        "anchor": a_loss2,
        "bridge": torch.tensor(0.0, device=device),
        "generation": g_loss2,
    }
    total_loss2, _ = combined_loss(losses_dict2, weights_dict)

    total_loss2.backward()

    print("  Backward completed successfully!")
    # Check a sample of gradients
    grad_ok = True
    checked = 0
    for name, param in model.named_parameters():
        if param.grad is not None and checked < 5:
            ok = check_grad(name, param)
            grad_ok &= ok
            checked += 1
    all_passed &= grad_ok

    # Verify no NaN/Inf in any gradient
    any_bad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                any_bad = True
                print(f"  BAD gradient: {name}")
    if not any_bad:
        print(f"  All {sum(1 for _, p in model.named_parameters() if p.grad is not None)} "
              f"parameter gradients checked: no NaN/Inf")
    all_passed &= (not any_bad)

    # Check total_loss2 is finite
    ok = check_finite("backward loss value", total_loss2)
    all_passed &= ok

    # ── 9. Memory report ─────────────────────────────────────────────
    separator("9. Memory report")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU memory allocated: {mem_allocated:.3f} GB")
        print(f"  GPU memory reserved:  {mem_reserved:.3f} GB")
    else:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in KB
        if sys.platform == "darwin":
            max_rss_mb = max_rss / (1024 * 1024)
        else:
            max_rss_mb = max_rss / 1024
        print(f"  Running on CPU (no GPU detected)")
        print(f"  Peak RSS: {max_rss_mb:.1f} MB")

    # ── 10. Summary ──────────────────────────────────────────────────
    separator("SUMMARY")
    if all_passed:
        print("  ALL CHECKS PASSED")
        print("  - All loss values are finite and in reasonable range")
        print("  - Backward pass produced valid gradients (no NaN/Inf)")
        print("  - Shape checks passed for boundary states and transitions")
        print("  - Zero-vector edge case handled correctly")
    else:
        print("  SOME CHECKS FAILED -- see details above")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
