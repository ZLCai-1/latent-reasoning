"""
Evaluator for Latent Reasoning.

Provides end-to-end evaluation: generation → answer extraction → metric
computation, with support for batch processing and multiple metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_accuracy, compute_exact_match, extract_numeric_answer

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a latent-reasoning model on a dataset.

    Runs model generation, extracts numeric answers, and computes
    configured metrics.

    Args:
        model: The :class:`LatentReasoningModel` to evaluate.
        dataloader: Evaluation data loader.
        metrics: List of metric names to compute (``"accuracy"``,
                 ``"exact_match"``).
        max_new_tokens: Maximum tokens to generate per sample.
    """

    def __init__(
        self,
        model: Any,
        dataloader: DataLoader,
        metrics: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        cot_baseline_tokens: int = 200,
        use_chat_template: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics or ["accuracy", "exact_match"]
        self.max_new_tokens = max_new_tokens
        self.cot_baseline_tokens = cot_baseline_tokens
        self.device = next(model.parameters()).device
        # Allow explicit override of chat template detection
        self._force_chat_template = use_chat_template

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metric results.

        Returns:
            Dictionary mapping metric names to values, including efficiency
            metrics: ``avg_tokens``, ``avg_latency_ms``, ``token_reduction``.
        """
        self.model.eval()

        all_predictions: List[str] = []
        all_references: List[str] = []
        all_num_generated_tokens: List[int] = []
        all_inference_times: List[float] = []

        # Detect if model supports chat template (e.g. Qwen, Llama-Instruct)
        if self._force_chat_template is not None:
            use_chat_template = self._force_chat_template
        else:
            use_chat_template = (
                hasattr(self.model.tokenizer, 'chat_template')
                and self.model.tokenizer.chat_template is not None
            )

        # Find "Answer:" token ids for prompt truncation (non-chat models)
        answer_prefix_ids = self.model.tokenizer.encode(
            "Answer:", add_special_tokens=False
        )

        for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]
            questions = batch.get("question", [])  # raw question strings

            batch_size = input_ids.size(0)

            # Check if this is student mode (has latent_positions)
            has_latent = batch.get("latent_positions") is not None
            latent_token_id = getattr(self.model, 'latent_token_id', None)

            if use_chat_template and questions:
                # Chat-template models: build prompt from raw question
                prompt_input_ids_list = []
                prompt_mask_list = []
                new_latent_positions = []  # recalculated for new prompt

                # Student mode: direct answer; Teacher mode: full CoT
                if has_latent:
                    system_content = "Give only the final numerical answer within \\boxed{}. Do not explain."
                else:
                    system_content = "Please reason step by step, and put your final answer within \\boxed{}."

                for i in range(batch_size):
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": questions[i]},
                    ]
                    prompt_text = self.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    encoded = self.model.tokenizer(
                        prompt_text, return_tensors="pt", add_special_tokens=False
                    )
                    p_ids = encoded["input_ids"].squeeze(0).to(self.device)
                    p_mask = encoded["attention_mask"].squeeze(0).to(self.device)

                    # For student mode: append <LATENT> tokens to prompt
                    if has_latent and latent_token_id is not None:
                        num_latent = self.model.num_latent_tokens
                        latent_ids = torch.full((num_latent,), latent_token_id, device=self.device)
                        latent_mask = torch.ones(num_latent, device=self.device)
                        # Record latent positions (relative to new prompt)
                        start_pos = p_ids.size(0)
                        new_latent_positions.append(
                            torch.arange(start_pos, start_pos + num_latent, device=self.device)
                        )
                        p_ids = torch.cat([p_ids, latent_ids])
                        p_mask = torch.cat([p_mask, latent_mask])

                    prompt_input_ids_list.append(p_ids)
                    prompt_mask_list.append(p_mask)

                # Override latent_positions with recalculated ones
                if new_latent_positions:
                    batch["latent_positions"] = torch.stack(new_latent_positions)
            else:
                # Non-chat models: truncate to "Answer:" prefix
                # In student mode, also keep <LATENT> tokens after "Answer:"
                latent_token_id = getattr(self.model, 'latent_token_id', None)
                prompt_input_ids_list = []
                prompt_mask_list = []
                for i in range(batch_size):
                    ids = input_ids[i].tolist()
                    cut_pos = self._find_answer_prefix(ids, answer_prefix_ids)
                    if cut_pos is not None:
                        end = cut_pos + len(answer_prefix_ids)
                        # Keep <LATENT> tokens (student mode)
                        if latent_token_id is not None:
                            while end < len(ids) and ids[end] == latent_token_id:
                                end += 1
                    else:
                        end = len(ids) // 3
                    prompt_input_ids_list.append(input_ids[i, :end])
                    prompt_mask_list.append(attention_mask[i, :end])

            # Determine padding side based on model type
            model_type = getattr(self.model.model.config, 'model_type', 'gpt2')
            pad_side = "right" if model_type == "gpt2" else "left"

            # Pad prompts to same length for batched generation
            max_prompt_len = max(p.size(0) for p in prompt_input_ids_list)
            pad_id = self.model.tokenizer.pad_token_id or 0
            padded_ids = []
            padded_mask = []
            pad_lengths = []  # Track padding for latent_positions offset
            for p_ids, p_mask in zip(prompt_input_ids_list, prompt_mask_list):
                pad_len = max_prompt_len - p_ids.size(0)
                pad_lengths.append(pad_len)
                pad_tensor = torch.full((pad_len,), pad_id, device=self.device)
                mask_pad = torch.zeros(pad_len, device=self.device)
                if pad_side == "left":
                    padded_ids.append(torch.cat([pad_tensor, p_ids]))
                    padded_mask.append(torch.cat([mask_pad, p_mask]))
                else:
                    padded_ids.append(torch.cat([p_ids, pad_tensor]))
                    padded_mask.append(torch.cat([p_mask, mask_pad]))
            prompt_input_ids = torch.stack(padded_ids).long()
            prompt_attention_mask = torch.stack(padded_mask).long()

            # Generate with timing
            t_start = time.time()
            # Recompute latent_positions from actual padded prompt
            # (batch's latent_positions don't account for collate_fn + evaluator padding)
            latent_pos = None
            latent_token_id = getattr(self.model, 'latent_token_id', None)
            if latent_token_id is not None and has_latent:
                new_latent_pos = []
                for i in range(batch_size):
                    ids = prompt_input_ids[i].tolist()
                    positions = [j for j, tid in enumerate(ids) if tid == latent_token_id]
                    if positions:
                        new_latent_pos.append(
                            torch.tensor(positions[:self.model.num_latent_tokens], device=self.device)
                        )
                if new_latent_pos and len(new_latent_pos) == batch_size:
                    latent_pos = torch.stack(new_latent_pos)
            generated_ids = self.model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                latent_positions=latent_pos,
                max_new_tokens=self.max_new_tokens,
            )
            t_end = time.time()
            batch_time = t_end - t_start
            per_sample_time = batch_time / batch_size

            # Decode predictions (only the generated part)
            answers = batch.get("answer", [])  # raw answer strings
            for i in range(batch_size):
                prompt_len = prompt_input_ids.size(1)
                gen_tokens = generated_ids[i, prompt_len:]
                # Count non-padding generated tokens
                num_gen_tokens = int((gen_tokens != self.model.tokenizer.pad_token_id).sum().item()) if self.model.tokenizer.pad_token_id is not None else len(gen_tokens)
                all_num_generated_tokens.append(num_gen_tokens)
                all_inference_times.append(per_sample_time)

                pred_text = self.model.tokenizer.decode(
                    gen_tokens, skip_special_tokens=True
                )
                all_predictions.append(pred_text)

                # Use raw answer field as reference (avoids tokenizer decode issues)
                if answers:
                    all_references.append(str(answers[i]))
                else:
                    # Fallback: decode from labels
                    label_ids = labels[i]
                    valid_ids = label_ids[label_ids != -100]
                    ref_text = self.model.tokenizer.decode(
                        valid_ids, skip_special_tokens=True
                    )
                    all_references.append(ref_text)

        # Compute metrics
        results: Dict[str, float] = {}

        if "accuracy" in self.metrics:
            acc = compute_accuracy(all_predictions, all_references)
            results["accuracy"] = acc
            logger.info("Accuracy: %.4f", acc)

        if "exact_match" in self.metrics:
            em = compute_exact_match(all_predictions, all_references)
            results["exact_match"] = em
            logger.info("Exact Match: %.4f", em)

        results["num_samples"] = float(len(all_predictions))

        # Efficiency metrics
        avg_tokens = sum(all_num_generated_tokens) / max(len(all_num_generated_tokens), 1)
        avg_latency_s = sum(all_inference_times) / max(len(all_inference_times), 1)
        avg_latency_ms = avg_latency_s * 1000.0

        # CoT token reduction: assume explicit CoT baseline ~200 tokens
        cot_baseline_tokens = self.cot_baseline_tokens
        token_reduction = 1.0 - (avg_tokens / cot_baseline_tokens) if cot_baseline_tokens > 0 else 0.0

        results["avg_tokens"] = avg_tokens
        results["avg_latency_ms"] = avg_latency_ms
        results["token_reduction"] = token_reduction
        results["cot_baseline_tokens"] = float(cot_baseline_tokens)

        logger.info("Avg Output Tokens: %.1f (vs CoT baseline ~%d)", avg_tokens, cot_baseline_tokens)
        logger.info("Token Reduction: %.1f%%", token_reduction * 100)
        logger.info("Avg Latency: %.1fms/sample", avg_latency_ms)

        return results

    @staticmethod
    def _find_answer_prefix(ids: List[int], prefix_ids: List[int]) -> Optional[int]:
        """Find the starting position of 'Answer:' token sequence in ids."""
        prefix_len = len(prefix_ids)
        for i in range(len(ids) - prefix_len + 1):
            if ids[i:i + prefix_len] == prefix_ids:
                return i
        return None

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 5,
    ) -> List[Dict[str, str]]:
        """Generate a few samples for qualitative inspection.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            List of dicts with ``"input"``, ``"prediction"``,
            ``"reference"`` keys.
        """
        self.model.eval()
        samples: List[Dict[str, str]] = []

        # Detect chat template support (respect override)
        if self._force_chat_template is not None:
            use_chat_template = self._force_chat_template
        else:
            use_chat_template = (
                hasattr(self.model.tokenizer, 'chat_template')
                and self.model.tokenizer.chat_template is not None
            )

        # Find "Answer:" token ids for prompt truncation (non-chat models)
        answer_prefix_ids = self.model.tokenizer.encode(
            "Answer:", add_special_tokens=False
        )

        # Determine padding side
        model_type = getattr(self.model.model.config, 'model_type', 'gpt2')
        pad_side = "right" if model_type == "gpt2" else "left"

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]
            questions = batch.get("question", [])
            batch_size = input_ids.size(0)

            # Check if this is student mode (has latent_positions)
            has_latent = batch.get("latent_positions") is not None
            latent_token_id = getattr(self.model, 'latent_token_id', None)

            if use_chat_template and questions:
                prompt_input_ids_list = []
                prompt_mask_list = []
                new_latent_positions = []

                # Student mode: direct answer; Teacher mode: full CoT
                if has_latent:
                    system_content = "Give only the final numerical answer within \\boxed{}. Do not explain."
                else:
                    system_content = "Please reason step by step, and put your final answer within \\boxed{}."

                for i in range(batch_size):
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": questions[i]},
                    ]
                    prompt_text = self.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    encoded = self.model.tokenizer(
                        prompt_text, return_tensors="pt", add_special_tokens=False
                    )
                    p_ids = encoded["input_ids"].squeeze(0).to(self.device)
                    p_mask = encoded["attention_mask"].squeeze(0).to(self.device)

                    # For student mode: append <LATENT> tokens to prompt
                    if has_latent and latent_token_id is not None:
                        num_latent = self.model.num_latent_tokens
                        latent_ids = torch.full((num_latent,), latent_token_id, device=self.device)
                        latent_mask = torch.ones(num_latent, device=self.device)
                        start_pos = p_ids.size(0)
                        new_latent_positions.append(
                            torch.arange(start_pos, start_pos + num_latent, device=self.device)
                        )
                        p_ids = torch.cat([p_ids, latent_ids])
                        p_mask = torch.cat([p_mask, latent_mask])

                    prompt_input_ids_list.append(p_ids)
                    prompt_mask_list.append(p_mask)

                # Override latent_positions with recalculated ones
                if new_latent_positions:
                    batch["latent_positions"] = torch.stack(new_latent_positions)
            else:
                prompt_input_ids_list = []
                prompt_mask_list = []
                for i in range(batch_size):
                    ids = input_ids[i].tolist()
                    cut_pos = self._find_answer_prefix(ids, answer_prefix_ids)
                    if cut_pos is not None:
                        end = cut_pos + len(answer_prefix_ids)
                        # Keep <LATENT> tokens (student mode)
                        if latent_token_id is not None:
                            while end < len(ids) and ids[end] == latent_token_id:
                                end += 1
                    else:
                        end = len(ids) // 3
                    prompt_input_ids_list.append(input_ids[i, :end])
                    prompt_mask_list.append(attention_mask[i, :end])

            # Pad prompts for batched generation
            max_prompt_len = max(p.size(0) for p in prompt_input_ids_list)
            pad_id = self.model.tokenizer.pad_token_id or 0
            padded_ids = []
            padded_mask = []
            pad_lengths = []  # Track padding for latent_positions offset
            for p_ids, p_mask in zip(prompt_input_ids_list, prompt_mask_list):
                pad_len = max_prompt_len - p_ids.size(0)
                pad_lengths.append(pad_len)
                pad_tensor = torch.full((pad_len,), pad_id, device=self.device)
                mask_pad = torch.zeros(pad_len, device=self.device)
                if pad_side == "left":
                    padded_ids.append(torch.cat([pad_tensor, p_ids]))
                    padded_mask.append(torch.cat([mask_pad, p_mask]))
                else:
                    padded_ids.append(torch.cat([p_ids, pad_tensor]))
                    padded_mask.append(torch.cat([p_mask, mask_pad]))
            prompt_input_ids = torch.stack(padded_ids).long()
            prompt_attention_mask = torch.stack(padded_mask).long()

            # Pass latent_positions to generate for proper embedding injection
            # Recompute from actual padded prompt (robust to any padding scheme)
            latent_pos = None
            latent_token_id = getattr(self.model, 'latent_token_id', None)
            has_latent = batch.get("latent_positions") is not None
            if latent_token_id is not None and has_latent:
                new_latent_pos = []
                for i in range(batch_size):
                    ids = prompt_input_ids[i].tolist()
                    positions = [j for j, tid in enumerate(ids) if tid == latent_token_id]
                    if positions:
                        new_latent_pos.append(
                            torch.tensor(positions[:self.model.num_latent_tokens], device=self.device)
                        )
                if new_latent_pos and len(new_latent_pos) == batch_size:
                    latent_pos = torch.stack(new_latent_pos)
            generated_ids = self.model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                latent_positions=latent_pos,
                max_new_tokens=self.max_new_tokens,
            )

            for i in range(batch_size):
                if len(samples) >= num_samples:
                    return samples

                # Decode prompt as input text
                input_text = self.model.tokenizer.decode(
                    prompt_input_ids_list[i], skip_special_tokens=True
                )
                prompt_len = prompt_input_ids.size(1)
                gen_tokens = generated_ids[i, prompt_len:]
                pred_text = self.model.tokenizer.decode(
                    gen_tokens, skip_special_tokens=True
                )

                # Use raw answer as reference if available
                answers = batch.get("answer", [])
                if answers:
                    ref_text = str(answers[i])
                else:
                    label_ids = labels[i]
                    valid_ids = label_ids[label_ids != -100]
                    ref_text = self.model.tokenizer.decode(
                        valid_ids, skip_special_tokens=True
                    )

                samples.append(
                    {
                        "input": input_text,
                        "prediction": pred_text,
                        "reference": ref_text,
                    }
                )

        return samples
