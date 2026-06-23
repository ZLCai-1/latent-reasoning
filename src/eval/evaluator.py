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
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics or ["accuracy", "exact_match"]
        self.max_new_tokens = max_new_tokens
        self.cot_baseline_tokens = cot_baseline_tokens
        self.device = next(model.parameters()).device

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

        # Find "Answer:" token ids for prompt truncation
        answer_prefix_ids = self.model.tokenizer.encode(
            "Answer:", add_special_tokens=False
        )

        for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            batch_size = input_ids.size(0)

            # Truncate input to prompt only (up to and including "Answer:")
            # so the model generates the answer from scratch
            prompt_input_ids_list = []
            prompt_mask_list = []
            for i in range(batch_size):
                ids = input_ids[i].tolist()
                # Find "Answer:" position
                cut_pos = self._find_answer_prefix(ids, answer_prefix_ids)
                if cut_pos is not None:
                    # Keep up to and including "Answer:" tokens
                    end = cut_pos + len(answer_prefix_ids)
                else:
                    # Fallback: use first 1/3 as prompt
                    end = len(ids) // 3
                prompt_input_ids_list.append(input_ids[i, :end])
                prompt_mask_list.append(attention_mask[i, :end])

            # Pad prompts to same length for batched generation
            max_prompt_len = max(p.size(0) for p in prompt_input_ids_list)
            pad_id = self.model.tokenizer.pad_token_id or 0
            padded_ids = []
            padded_mask = []
            for p_ids, p_mask in zip(prompt_input_ids_list, prompt_mask_list):
                pad_len = max_prompt_len - p_ids.size(0)
                # Right-pad prompts (GPT-2 absolute position encoding)
                padded_ids.append(
                    torch.cat([p_ids, torch.full((pad_len,), pad_id, device=self.device)])
                )
                padded_mask.append(
                    torch.cat([p_mask, torch.zeros(pad_len, device=self.device)])
                )
            prompt_input_ids = torch.stack(padded_ids).long()
            prompt_attention_mask = torch.stack(padded_mask).long()

            # Generate with timing
            t_start = time.time()
            generated_ids = self.model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_new_tokens,
            )
            t_end = time.time()
            batch_time = t_end - t_start
            per_sample_time = batch_time / batch_size

            # Decode predictions (only the generated part)
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

                # Decode reference from labels
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

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
            )

            for i in range(generated_ids.size(0)):
                if len(samples) >= num_samples:
                    return samples

                input_text = self.model.tokenizer.decode(
                    input_ids[i], skip_special_tokens=True
                )
                prompt_len = input_ids.size(1)
                gen_tokens = generated_ids[i, prompt_len:]
                pred_text = self.model.tokenizer.decode(
                    gen_tokens, skip_special_tokens=True
                )
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
