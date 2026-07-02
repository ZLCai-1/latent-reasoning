#!/usr/bin/env python3
"""
Run full diagnostic evaluation for latent reasoning models.

Produces a structured report with all §5.6 metrics from the research proposal:
- §5.6.2 Task performance (accuracy, retention, length-bucket)
- §5.6.3 Efficiency (token reduction, latency, throughput, compression ratio)
- §5.6.4 State transition alignment (transition cosine, NTE, endpoint drift, CKA)
- §5.6.5 Stability & interpretability (collapse rate, diversity, effective rank)

Usage:
    python scripts/run_diagnostics.py \
        --config config/exp/stage1_transition.yaml \
        --checkpoint checkpoints/stage1_transition/final \
        --data_path data/gsm8k_test.json \
        --output results/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LatentReasoningDataset, collate_fn, load_gsm8k
from src.data.state_extractor import TeacherStateExtractor
from src.eval.evaluator import Evaluator
from src.eval.diagnostics import (
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
)
from src.models.base import LatentReasoningModel
from src.models.state_transition import StateTransitionModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diagnostic Evaluation")
    parser.add_argument("--config", type=str, default="config/exp/stage1_transition.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all, set small (e.g. 10) for quick sanity check")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Number of qualitative samples to save in output")
    parser.add_argument("--cot_accuracy", type=float, default=0.0,
                        help="Explicit CoT accuracy for retention calc (0=skip)")
    parser.add_argument("--direct_accuracy", type=float, default=0.0,
                        help="Direct answer accuracy for gain calc (0=skip)")
    parser.add_argument("--cot_avg_tokens", type=float, default=200.0,
                        help="Avg CoT output tokens for compression ratio")
    parser.add_argument("--output", type=str, default="results/diagnostics.json",
                        help="Path to save results JSON")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Disable chat template for student evaluation (use raw text format matching training)")
    # --- Latent token intervention args (Stage 0.3) ---
    parser.add_argument("--zero_latent", action="store_true",
                        help="Zero out latent embeddings before evaluation")
    parser.add_argument("--random_latent", action="store_true",
                        help="Replace latent embeddings with random values")
    parser.add_argument("--shuffle_latent", action="store_true",
                        help="Shuffle the order of latent embeddings")
    parser.add_argument("--repeat_latent", action="store_true",
                        help="Use first latent token for all positions")
    return parser.parse_args()


def load_model_and_config(args):
    """Load config and model (with LoRA support)."""
    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        base_refs = cfg.pop("defaults")
        config_dir = os.path.dirname(os.path.abspath(args.config))
        for ref in base_refs:
            base_path = os.path.join(config_dir, ref + ".yaml")
            if os.path.exists(base_path):
                base_cfg = OmegaConf.load(base_path)
                cfg = OmegaConf.merge(base_cfg, cfg)

    model_cfg = cfg.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint
    base_model_name = model_cfg.get("name", "gpt2")
    num_latent_tokens = model_cfg.get("num_latent_tokens", 0)

    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    if is_lora:
        model = LatentReasoningModel(
            model_name=base_model_name,
            layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
            num_latent_tokens=num_latent_tokens,
            device=device,
        )
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, checkpoint_path)
        model.model.eval()
        # Load latent embeddings
        parent = os.path.dirname(checkpoint_path)
        ckpt_files = sorted([f for f in os.listdir(parent) if f.startswith('checkpoint_epoch') and f.endswith('.pt')])
        if ckpt_files:
            ckpt_path = os.path.join(parent, ckpt_files[-1])
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt and 'latent_embeddings.weight' in ckpt['model_state_dict']:
                model.latent_embeddings.weight.data = ckpt['model_state_dict']['latent_embeddings.weight']
    else:
        model = LatentReasoningModel(
            model_name=checkpoint_path,
            layer_ids=list(model_cfg.get("layer_ids", [-1, -2])),
            num_latent_tokens=num_latent_tokens,
            device=device,
        )

    return model, cfg


def run_task_metrics(model, cfg, args):
    """§5.6.2 & §5.6.3: Task performance + efficiency metrics."""
    logger.info("=" * 60)
    logger.info("§5.6.2 & §5.6.3: Task Performance & Efficiency")
    logger.info("=" * 60)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    num_latent_tokens = model_cfg.get("num_latent_tokens", 0)

    data_path = args.data_path or data_cfg.get("data_path")
    raw_data = load_gsm8k(data_path=data_path, split=args.split)
    if args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
        mode="student" if num_latent_tokens > 0 else "teacher",
        num_latent_tokens=num_latent_tokens,
    )

    pad_token_id = model.tokenizer.pad_token_id or 0
    model_type = getattr(model.model.config, 'model_type', 'gpt2')
    eval_padding_side = "right" if model_type == "gpt2" else "left"
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id, padding_side=eval_padding_side),
    )

    evaluator = Evaluator(
        model=model, dataloader=dataloader,
        metrics=["accuracy", "exact_match"],
        max_new_tokens=args.max_new_tokens,
        cot_baseline_tokens=int(args.cot_avg_tokens),
        use_chat_template=False if args.no_chat_template else None,
    )

    t_start = time.time()
    results = evaluator.evaluate()
    total_time = time.time() - t_start

    # Extended metrics
    accuracy = results.get("accuracy", 0.0)
    results["throughput_samples_per_sec"] = compute_throughput(len(raw_data), total_time)
    results["compression_ratio"] = compute_compression_ratio(args.cot_avg_tokens, num_latent_tokens)

    if args.cot_accuracy > 0:
        results["accuracy_retention"] = compute_accuracy_retention(accuracy, args.cot_accuracy)
    if args.direct_accuracy > 0:
        results["relative_gain_over_direct"] = compute_relative_gain(accuracy, args.direct_accuracy)

    return results


def run_transition_diagnostics(model, cfg, args):
    """§5.6.4: State transition alignment metrics."""
    logger.info("=" * 60)
    logger.info("§5.6.4: State Transition Alignment Diagnostics")
    logger.info("=" * 60)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})
    num_latent_tokens = model_cfg.get("num_latent_tokens", 0)
    layer_ids = list(model_cfg.get("layer_ids", [-1, -2]))

    # Load teacher states
    cache_dir = teacher_cfg.get("cache_dir", "data/teacher_states")
    cache_file = os.path.join(cache_dir, "teacher_states.h5")
    if not os.path.exists(cache_file):
        logger.warning("Teacher state cache not found at %s — skipping §5.6.4", cache_file)
        return {}

    teacher_states = TeacherStateExtractor.load_cached_states(cache_file, layer_ids, device="cpu")

    # Load evaluation data
    data_path = args.data_path or data_cfg.get("data_path")
    raw_data = load_gsm8k(data_path=data_path, split=args.split)
    if args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]

    dataset = LatentReasoningDataset(
        data=raw_data,
        tokenizer=model.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 512),
        num_spans=data_cfg.get("num_spans", 3),
        span_strategy=data_cfg.get("span_strategy", "fixed"),
        mode="student" if num_latent_tokens > 0 else "teacher",
        num_latent_tokens=num_latent_tokens,
    )

    pad_token_id = model.tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
    )

    device = next(model.parameters()).device
    num_layers = model.get_num_layers()
    resolved_ids = [lid if lid >= 0 else num_layers + 1 + lid for lid in layer_ids]
    transition_module = StateTransitionModule(layer_ids=resolved_ids).to(device)

    # Collect student transitions and boundary states
    all_student_trans = []
    all_teacher_trans = []
    all_student_boundary = []
    all_teacher_boundary = []
    all_latent_states = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing transitions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latent_positions = batch.get("latent_positions")
            sample_indices = batch.get("sample_idx")

            if latent_positions is None:
                continue
            latent_positions = latent_positions.to(device)

            # Student forward with latent injection
            outputs = model.forward_with_latent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_positions=latent_positions,
                output_hidden_states=True,
            )

            hidden_states = outputs["hidden_states"]

            # Extract student boundary states at latent positions
            student_boundary = StateTransitionModule.extract_boundary_states(
                hidden_states, latent_positions, resolved_ids
            )  # [B, K, nL, D]
            student_trans = StateTransitionModule.compute_transitions(student_boundary)

            # Get teacher transitions for same samples
            if sample_indices is not None:
                indices = sample_indices.tolist()
                boundary_dict = teacher_states.get("boundary_states", {})
                if boundary_dict:
                    layer_tensors = list(boundary_dict.values())
                    if layer_tensors:
                        t_boundary = torch.stack(layer_tensors, dim=2)  # [N, K_all, nL, D]
                        t_boundary_end = t_boundary[:, 1::2, :, :]  # SPAN_END only
                        valid = [i for i in indices if i < t_boundary_end.size(0)]
                        if valid:
                            t_batch = t_boundary_end[valid]  # [B', K, nL, D]
                            t_trans = StateTransitionModule.compute_transitions(t_batch)

                            # Align batch size
                            min_b = min(student_trans.size(0), t_trans.size(0))
                            all_student_trans.append(student_trans[:min_b].cpu())
                            all_teacher_trans.append(t_trans[:min_b].cpu())
                            all_student_boundary.append(student_boundary[:min_b].cpu())
                            all_teacher_boundary.append(t_batch[:min_b].cpu())

            # Collect latent hidden states for collapse analysis
            all_latent_states.append(student_boundary.cpu())

    if not all_student_trans:
        logger.warning("No valid transition data collected — skipping §5.6.4")
        return {}

    # Concatenate
    s_trans = torch.cat(all_student_trans, dim=0)
    t_trans = torch.cat(all_teacher_trans, dim=0)
    s_bound = torch.cat(all_student_boundary, dim=0)
    t_bound = torch.cat(all_teacher_boundary, dim=0)
    latent_states = torch.cat(all_latent_states, dim=0)

    # Compute §5.6.4 metrics
    results = {}
    results["transition_cosine"] = transition_cosine(s_trans, t_trans)
    results["normalized_transition_error"] = normalized_transition_error(s_trans, t_trans)
    results["endpoint_drift"] = endpoint_drift(s_bound, t_bound)
    results["layer_wise_cka"] = layer_wise_cka(s_bound, t_bound)

    # §5.6.5 collapse metrics
    results["collapse_metrics"] = collapse_metrics(latent_states)

    return results


def print_report(task_results, diag_results):
    """Print a formatted diagnostic report."""
    print("\n")
    print("=" * 70)
    print("         LATENT REASONING DIAGNOSTIC REPORT")
    print("=" * 70)

    # §5.6.2 Task Performance
    print("\n┌─── §5.6.2 Task Performance ───────────────────────────────────────┐")
    if "accuracy" in task_results:
        print(f"│  Accuracy:              {task_results['accuracy'] * 100:.2f}%")
    if "exact_match" in task_results:
        print(f"│  Exact Match:           {task_results['exact_match'] * 100:.2f}%")
    if "accuracy_retention" in task_results:
        print(f"│  Accuracy Retention:    {task_results['accuracy_retention']:.4f}")
    if "relative_gain_over_direct" in task_results:
        print(f"│  Relative Gain/Direct:  {task_results['relative_gain_over_direct']:.4f}")
    print("└───────────────────────────────────────────────────────────────────┘")

    # §5.6.3 Efficiency
    print("\n┌─── §5.6.3 Efficiency ─────────────────────────────────────────────┐")
    if "avg_tokens" in task_results:
        print(f"│  Avg Output Tokens:     {task_results['avg_tokens']:.1f}")
    if "token_reduction" in task_results:
        print(f"│  Token Reduction:       {task_results['token_reduction'] * 100:.1f}%")
    if "compression_ratio" in task_results:
        print(f"│  Compression Ratio:     {task_results['compression_ratio']:.1f}x")
    if "avg_latency_ms" in task_results:
        print(f"│  Avg Latency:           {task_results['avg_latency_ms']:.1f} ms/sample")
    if "throughput_samples_per_sec" in task_results:
        print(f"│  Throughput:            {task_results['throughput_samples_per_sec']:.2f} samples/sec")
    print("└───────────────────────────────────────────────────────────────────┘")

    # §5.6.4 Transition Alignment
    if diag_results:
        print("\n┌─── §5.6.4 State Transition Alignment ─────────────────────────────┐")
        if "transition_cosine" in diag_results:
            tc = diag_results["transition_cosine"]
            print(f"│  Transition Cosine:     {tc['mean']:.4f}  (↑ better, max=1.0)")
            if "per_k" in tc:
                per_k_str = ", ".join([f"k{i}={v:.3f}" for i, v in enumerate(tc["per_k"])])
                print(f"│    per-k: {per_k_str}")
            if "per_layer" in tc:
                per_l_str = ", ".join([f"L{i}={v:.3f}" for i, v in enumerate(tc["per_layer"])])
                print(f"│    per-layer: {per_l_str}")

        if "normalized_transition_error" in diag_results:
            nte = diag_results["normalized_transition_error"]
            print(f"│  Norm. Transition Err:  {nte['mean']:.4f}  (↓ better)")

        if "endpoint_drift" in diag_results:
            ed = diag_results["endpoint_drift"]
            print(f"│  Endpoint Drift:        {ed['mean']:.4f}  (↓ better)")

        if "layer_wise_cka" in diag_results:
            cka = diag_results["layer_wise_cka"]
            print(f"│  Layer-wise CKA (mean): {cka['mean']:.4f}  (↑ better)")
            if "per_layer" in cka:
                per_l_str = ", ".join([f"L{i}={v:.3f}" for i, v in enumerate(cka["per_layer"])])
                print(f"│    per-layer: {per_l_str}")
        print("└───────────────────────────────────────────────────────────────────┘")

        # §5.6.5 Stability
        print("\n┌─── §5.6.5 Stability & Collapse ──────────────────────────────────┐")
        if "collapse_metrics" in diag_results:
            cm = diag_results["collapse_metrics"]
            print(f"│  Pairwise Diversity:    {cm['pairwise_diversity']:.4f}  (↑ better)")
            print(f"│  Mean Cosine:           {cm['mean_cosine']:.4f}")
            print(f"│  Collapse Rate:         {cm['collapse_rate'] * 100:.1f}%  (↓ better)")
            print(f"│  Effective Rank:        {cm['effective_rank']:.2f}  (↑ better)")
        print("└───────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # Load model
    model, cfg = load_model_and_config(args)

    # --- Apply latent token interventions (Stage 0.3) ---
    if args.zero_latent and model.num_latent_tokens > 0:
        logger.info("[Intervention] Zeroing latent embeddings")
        model.latent_embeddings.weight.data.zero_()
    elif args.random_latent and model.num_latent_tokens > 0:
        logger.info("[Intervention] Replacing latent embeddings with random values")
        torch.manual_seed(12345)
        nn.init.normal_(model.latent_embeddings.weight, mean=0.0, std=0.02)
    elif args.shuffle_latent and model.num_latent_tokens > 0:
        logger.info("[Intervention] Shuffling latent embedding order")
        perm = torch.randperm(model.num_latent_tokens)
        model.latent_embeddings.weight.data = model.latent_embeddings.weight.data[perm]
    elif args.repeat_latent and model.num_latent_tokens > 0:
        logger.info("[Intervention] Repeating first latent token for all positions")
        first = model.latent_embeddings.weight.data[0:1].clone()
        model.latent_embeddings.weight.data = first.expand(model.num_latent_tokens, -1).contiguous()

    # Run task metrics (§5.6.2 & §5.6.3)
    task_results = run_task_metrics(model, cfg, args)

    # Run transition diagnostics (§5.6.4 & §5.6.5)
    diag_results = run_transition_diagnostics(model, cfg, args)

    # Generate qualitative samples for inspection
    samples = []
    if args.show_samples > 0:
        from src.eval.evaluator import Evaluator
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        num_latent_tokens = model_cfg.get("num_latent_tokens", 0)
        data_path = args.data_path or data_cfg.get("data_path")
        raw_data = load_gsm8k(data_path=data_path, split=args.split)
        if args.max_samples > 0:
            raw_data = raw_data[:args.max_samples]

        dataset = LatentReasoningDataset(
            data=raw_data,
            tokenizer=model.tokenizer,
            max_seq_length=data_cfg.get("max_seq_length", 512),
            num_spans=data_cfg.get("num_spans", 3),
            span_strategy=data_cfg.get("span_strategy", "fixed"),
            mode="student" if num_latent_tokens > 0 else "teacher",
            num_latent_tokens=num_latent_tokens,
        )
        pad_token_id = model.tokenizer.pad_token_id or 0
        model_type = getattr(model.model.config, 'model_type', 'gpt2')
        eval_padding_side = "right" if model_type == "gpt2" else "left"
        sample_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id, padding_side=eval_padding_side),
        )
        evaluator = Evaluator(
            model=model, dataloader=sample_loader,
            metrics=["accuracy"],
            max_new_tokens=args.max_new_tokens,
            use_chat_template=False if args.no_chat_template else None,
        )
        samples = evaluator.generate_samples(num_samples=args.show_samples)

        # Print samples to terminal
        print("\n" + "=" * 70)
        print("         QUALITATIVE SAMPLES")
        print("=" * 70)
        for i, s in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"  Question:   {s['input'][:150]}")
            print(f"  Prediction: {s['prediction'][:150]}")
            print(f"  Reference:  {s['reference']}")
        print("=" * 70)

    # Print formatted report
    print_report(task_results, diag_results)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    full_results = {
        "task_metrics": {k: v for k, v in task_results.items() if not isinstance(v, torch.Tensor)},
        "transition_diagnostics": diag_results,
        "qualitative_samples": samples,
    }
    with open(args.output, "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info("Full diagnostics saved to %s", args.output)


if __name__ == "__main__":
    main()
