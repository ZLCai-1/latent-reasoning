"""
Evaluator for Latent Reasoning.

Provides end-to-end evaluation: generation → answer extraction → metric
computation, with support for batch processing and multiple metrics.
"""

from __future__ import annotations

import logging
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
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics or ["accuracy", "exact_match"]
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metric results.

        Returns:
            Dictionary mapping metric names to values.
        """
        self.model.eval()

        all_predictions: List[str] = []
        all_references: List[str] = []

        for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            # Generate
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
            )

            # Decode predictions (only the generated part)
            for i in range(generated_ids.size(0)):
                prompt_len = input_ids.size(1)
                gen_tokens = generated_ids[i, prompt_len:]
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
        return results

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
