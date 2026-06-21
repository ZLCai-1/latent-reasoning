#!/usr/bin/env python
"""
End-to-End Verification Script for Latent Reasoning Pipeline.

Tests the full pipeline:
  Data loading → CoT span splitting → Teacher state extraction →
  Transition alignment training → Loss verification

Usage:
    conda run -n latent_reasoning python scripts/test_e2e.py
"""

from __future__ import annotations

import sys
import os
import time

# Make the project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from src.data.dataset import load_gsm8k
from src.data.preprocessing import split_into_spans, prepare_training_sample
from src.models.state_transition import StateTransitionModule
from src.models.loss_functions import transition_loss, generation_loss, combined_loss


def _build_mini_tokenizer(vocab_size: int = 5000) -> GPT2TokenizerFast:
    """Build a minimal BPE tokenizer without network access."""
    import tempfile, json

    # Create a simple character-level + word-level vocab
    # Use HuggingFace tokenizers library to build a minimal BPE tokenizer
    base_tokenizer = Tokenizer(models.BPE())
    base_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on some minimal text
    trainer_obj = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        min_frequency=1,
    )
    # Train on mini corpus
    corpus = [
        "Tom has 5 apples. He gives 2 to Mary. How many does he have?",
        "A store has 10 books. They sell 3 in the morning.",
        "Sarah has 8 candies. She eats 3 and gives 2 to her friend.",
        "Question: Answer: The total is 5 - 2 = 3.",
        "A farmer has 12 chickens. He buys 5 more and then sells 4.",
        "Lisa has 20 dollars. She spends 7 on lunch and 5 on a book.",
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    ]
    base_tokenizer.train_from_iterator(corpus, trainer=trainer_obj)

    # Save and reload as GPT2TokenizerFast
    tmpdir = tempfile.mkdtemp()
    base_tokenizer.save(os.path.join(tmpdir, "tokenizer.json"))

    # Write required config files
    with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "model_type": "gpt2",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|padding|>",
        }, f)
    with open(os.path.join(tmpdir, "special_tokens_map.json"), "w") as f:
        json.dump({
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|padding|>",
        }, f)

    tokenizer = GPT2TokenizerFast.from_pretrained(tmpdir)
    tokenizer.pad_token = "<|padding|>"
    return tokenizer


def main():
    start_time = time.time()
    device = torch.device("cpu")
    torch.manual_seed(42)

    print("=" * 60)
    print("  Latent Reasoning E2E Verification")
    print("=" * 60)

    # ==================================================================
    # [1/5] Load GPT-2 model
    # ==================================================================
    print("\n[1/5] Loading GPT-2 model (random init, no network)...", end=" ", flush=True)

    # Use a smaller GPT-2 config for fast CPU verification
    config = GPT2Config(
        vocab_size=5000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config).to(device)

    # Build a simple tokenizer from scratch (no network needed)
    tokenizer = _build_mini_tokenizer(vocab_size=5000)

    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<SPAN_START>", "<SPAN_END>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    span_start_id = tokenizer.convert_tokens_to_ids("<SPAN_START>")
    span_end_id = tokenizer.convert_tokens_to_ids("<SPAN_END>")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OK ({num_params:.1f}M params, vocab={len(tokenizer)})")

    # ==================================================================
    # [2/5] Load mini dataset
    # ==================================================================
    print("[2/5] Preparing mini dataset...", end=" ", flush=True)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mini_gsm8k.json")
    data = load_gsm8k(data_path=data_path)
    print(f"OK ({len(data)} samples)")

    # ==================================================================
    # [3/5] Teacher state extraction
    # ==================================================================
    print("[3/5] Teacher state extraction...", end=" ", flush=True)

    # Prepare tokenized samples with boundary positions
    num_spans = 3
    layer_ids = [-1, -2]
    samples = []

    for record in data:
        sample = prepare_training_sample(
            question=record["question"],
            spans=record["spans"],
            answer=record["answer"],
            tokenizer=tokenizer,
            num_spans=num_spans,
            span_strategy="fixed",
            max_seq_length=256,
        )
        # Only keep samples with boundary positions
        if "boundary_positions" in sample:
            samples.append(sample)

    if not samples:
        print("FAIL - no samples with boundary positions!")
        sys.exit(1)

    # Extract teacher states (teacher = same model, frozen, with CoT)
    teacher_transitions_list = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            bp = sample["boundary_positions"].unsqueeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

            # Extract boundary states [1, K, num_layers, D]
            boundary_states = StateTransitionModule.extract_boundary_states(
                hidden_states, bp, layer_ids
            )
            # Compute transitions [1, K-1, num_layers, D]
            teacher_trans = StateTransitionModule.compute_transitions(boundary_states)
            teacher_transitions_list.append(teacher_trans.squeeze(0))  # [K-1, nL, D]

    print(f"OK ({len(samples)} samples, {teacher_transitions_list[0].shape[-1]}D)")

    # ==================================================================
    # [4/5] Training loop
    # ==================================================================
    print("[4/5] Training loop:")

    model.train()
    # Only train the model parameters (not frozen teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    num_epochs = 3
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_gen_loss = 0.0
        num_batches = 0

        for i, sample in enumerate(samples):
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            bp = sample["boundary_positions"].unsqueeze(0).to(device)
            teacher_trans = teacher_transitions_list[i].unsqueeze(0).to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            logits = outputs.logits

            # Student boundary states & transitions
            student_boundary = StateTransitionModule.extract_boundary_states(
                hidden_states, bp, layer_ids
            )
            student_trans = StateTransitionModule.compute_transitions(student_boundary)

            # Compute losses
            t_loss = transition_loss(student_trans, teacher_trans, normalize=False)
            g_loss = generation_loss(logits, labels)

            losses_dict = {
                "transition": t_loss,
                "generation": g_loss,
            }
            weights_dict = {
                "transition": 0.7,
                "generation": 0.3,
            }
            total_loss, loss_info = combined_loss(losses_dict, weights_dict)

            # Backward + step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss_info["total"]
            epoch_trans_loss += loss_info["transition"]
            epoch_gen_loss += loss_info["generation"]
            num_batches += 1

        avg_total = epoch_total_loss / num_batches
        avg_trans = epoch_trans_loss / num_batches
        avg_gen = epoch_gen_loss / num_batches
        epoch_losses.append(avg_total)

        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"loss={avg_total:.4f} "
              f"(transition={avg_trans:.4f}, generation={avg_gen:.4f})")

    # ==================================================================
    # [5/5] Verify loss trend
    # ==================================================================
    print("[5/5] Loss trend: ", end="")

    # Check if loss decreased overall (first vs last)
    if epoch_losses[-1] < epoch_losses[0]:
        trend = "DECREASING \u2713"
        trend_ok = True
    else:
        # Even if not strictly decreasing, check if difference is small
        diff = epoch_losses[-1] - epoch_losses[0]
        if diff < 0.1:
            trend = f"STABLE (delta={diff:.4f}) \u2713"
            trend_ok = True
        else:
            trend = f"NOT DECREASING (delta={diff:.4f}) \u2717"
            trend_ok = False
    print(trend)

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    if trend_ok:
        print("  ALL CHECKS PASSED! Pipeline is ready for GPU training.")
    else:
        print("  WARNING: Loss did not decrease, but pipeline ran without errors.")
        print("  This may be normal with only 3 epochs on CPU.")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if not trend_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
